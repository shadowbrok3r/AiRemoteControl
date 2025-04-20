use anyhow::{anyhow, Context, Result};
// *** Using openai_responses SDK ***
use openai_responses::{
    // Main client and request/response types
    types::{
        config::Truncation,
        // Use SDK types based on list provided
        // *** Corrected import path for OutputItem, added InputItem ***
        item::{ClickButton, ComputerAction, ComputerCallOutput, ComputerToolCall, InputItem, OutputItem, SafetyCheck},
        // *** Added InputListItem, removed unused ContentItem, ImageDetail ***
        request::{Input, InputListItem, Request},
        tools::{Environment, Tool},
        Model,
    },
    Client as ResponsesClient,
};
// Keep rmcp imports for server interaction
use rmcp::{
    model::{CallToolRequestParam, RawContent}, // Removed ErrorCode, ErrorData
    serve_client,
    // *** Use Peer directly, ServiceRole needed for generic bound ***
    service::{RoleClient, RunningService},
    Peer, // Use Peer type directly
};
use serde::{Deserialize, Serialize};
use serde_json::Value; // Removed json macro import
// Removed std::env import
use tokio::net::TcpSocket;
use tracing::{debug, error, info, warn};

// Configuration
const MCP_SERVER_ADDR: &str = "127.0.0.1:9001";
const DISPLAY_WIDTH: u32 = 1920;
const DISPLAY_HEIGHT: u32 = 1080;
const ENVIRONMENT: Environment = Environment::Windows; // Use SDK Enum

// Helper struct to deserialize screenshot results from MCP server
#[derive(Deserialize, Debug)]
struct ScreenshotResultData {
    base64_data: Option<String>,
}

// Parameter structs for calling MCP execute_openai_* tools
#[derive(Debug, Serialize)] struct OpenAIClickParams { x: i32, y: i32, button: String }
#[derive(Debug, Serialize)] struct OpenAIScrollParams { x: i32, y: i32, scroll_x: i32, scroll_y: i32 }
#[derive(Debug, Serialize)] struct OpenAIKeyPressParams { keys: Vec<String> }
#[derive(Debug, Serialize)] struct OpenAITypeParams { text: String }
#[derive(Debug, Serialize)] struct OpenAIWaitParams { duration_ms: Option<u64> }
#[derive(Debug, Serialize)] struct CaptureScreenParams { x: Option<i32>, y: Option<i32>, width: Option<u32>, height: Option<u32> }


pub async fn run_computer_use() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env().add_directive(tracing::Level::INFO.into()))
        .with_writer(std::io::stdout)
        .with_ansi(true)
        .init();

    // Load OpenAI API Key
    dotenv::dotenv().ok();
    let openai_client = ResponsesClient::from_env()
        .context("Failed to create OpenAI Responses Client. Ensure OPENAI_API_KEY is set.")?;

    // --- Connect to MCP Server ---
    info!("Connecting to MCP Server at {}...", MCP_SERVER_ADDR);
    let stream = TcpSocket::new_v4()?
        .connect(MCP_SERVER_ADDR.parse()?)
        .await
        .context(format!("Failed to connect to MCP server at {}", MCP_SERVER_ADDR))?;
    info!("Connected to MCP Server.");

    // Start the MCP client service
    let mcp_client: RunningService<RoleClient, ()> = serve_client((), stream)
        .await
        .context("Failed to establish MCP client service (ensure 'client' feature is enabled for rmcp)")?;
    let mcp_peer = mcp_client.peer().clone();

    // --- Get Initial User Task ---
    println!("\nEnter the computer task you want the AI to perform (or type 'quit'):");
    let mut user_input = String::new();
    std::io::stdin().read_line(&mut user_input)?;
    let user_input = user_input.trim();

    if user_input.eq_ignore_ascii_case("quit") || user_input.is_empty() {
        return Ok(());
    }

    // --- Initial Request to Responses API ---
    info!("Sending initial task to OpenAI Computer Use model...");

    // Define the Computer Use tool for the request using SDK types
    // TODO: Verify Tool::ComputerUse variant name and fields
    let computer_tool = Tool::ComputerUse {
        display_width: DISPLAY_WIDTH as u64,
        display_height: DISPLAY_HEIGHT as u64,
        environment: ENVIRONMENT,
    };

    // Construct the initial request using SDK types
    // TODO: Verify Model enum variant for computer-use-preview
    let initial_request = Request {
        model: Model::ComputerUsePreview, // Assuming this variant exists
        // *** Use Input::Text variant, wrapped in vec! ***
        input: Input::Text(user_input.to_string()),
        tools: Some(vec![computer_tool.clone()]),
        instructions: None,
        previous_response_id: None,
        reasoning: None,
        truncation: Some(Truncation::Auto),
        ..Default::default()
    };

    // --- Main Computer Use Loop ---
    let mut current_request = initial_request;
    // Removed last_response_id, use response.id directly

    loop {
        debug!("Sending request...");
        // *** Removed type annotation, use ? directly ***
        let response = openai_client.create(current_request.clone()).await
            .context("OpenAI Responses API call failed")?
            .unwrap();
        // info!("Received response: {:?}", response);

        let current_response_id = response.id.clone();

        // Find the computer_call action in the response output
        let computer_call_opt: Option<&ComputerToolCall> = response.output.iter().find_map(|item| {
            if let OutputItem::ComputerToolCall(call) = item { Some(call) } else { None }
        });

        if let Some(computer_call) = computer_call_opt {
            let call_id = computer_call.call_id.clone();
            let action = &computer_call.action;

            info!("Received action type: {:?}", action);

            // --- Execute Action using MCP Server ---
            let execution_result = match action {
                ComputerAction::Click { x, y, button } => {
                    let button_str = match button {
                         ClickButton::Left=>"left",
                         ClickButton::Right=>"right",
                         ClickButton::Wheel=> "middle",
                         ClickButton::Back => "back",
                         ClickButton::Forward => "forward",
                    }.to_string();
                    let params = OpenAIClickParams { x: x.to_owned() as i32, y: y.to_owned() as i32, button: button_str };
                    call_mcp_tool(&mcp_peer, "execute_openai_click", params).await
                }
                ComputerAction::Scroll { x, y, scroll_x, scroll_y } => {
                    let params = OpenAIScrollParams { x: x.to_owned() as i32, y: y.to_owned() as i32, scroll_x: scroll_x.to_owned() as i32, scroll_y: scroll_y.to_owned() as i32 };
                    call_mcp_tool(&mcp_peer, "execute_openai_scroll", params).await
                }
                ComputerAction::KeyPress { keys } => {
                    let params = OpenAIKeyPressParams { keys: keys.clone() };
                    call_mcp_tool(&mcp_peer, "execute_openai_keypress", params).await
                }
                ComputerAction::Type { text } => {
                    let params = OpenAITypeParams { text: text.clone() };
                    call_mcp_tool(&mcp_peer, "execute_openai_type", params).await
                }
                 ComputerAction::Wait => {
                    let params = OpenAIWaitParams { duration_ms: None };
                    call_mcp_tool(&mcp_peer, "execute_openai_wait", params).await
                 }
                 ComputerAction::Screenshot => {
                     info!("Received Screenshot action (handled implicitly).");
                     Ok(())
                 }
                 ComputerAction::Move { x, y } => {
                     warn!("Received Move action. Mapping to execute_openai_click at ({}, {}) with no button press.", x, y);
                     let params = OpenAIClickParams { x: x.to_owned() as i32, y: y.to_owned() as i32, button: "none".to_string() };
                     call_mcp_tool(&mcp_peer, "execute_openai_click", params).await
                 }
                 ComputerAction::DoubleClick { x, y } => {
                    warn!("Received DoubleClick action. Mapping to single left click for now.");
                    let params = OpenAIClickParams { x: x.to_owned() as i32, y: y.to_owned() as i32, button: "left".to_string() };
                    call_mcp_tool(&mcp_peer, "execute_openai_click", params).await
                 }
                 ComputerAction::Drag { .. } => {
                     warn!("Received Drag action, which is not implemented yet.");
                     Ok(())
                 }
            };

            if let Err(e) = execution_result {
                error!("Failed to execute MCP tool for action '{:?}': {}", action, e);
                println!("Error executing action '{:?}'. Stopping.", action);
                break;
            }

            // --- Capture Screenshot ---
            info!("Capturing screen after action...");
            let screenshot_base64 = match call_capture_screen(&mcp_peer, None, None, None, None).await {
                 Ok(b64) => b64,
                 Err(e) => {
                     error!("Failed to capture screen: {}", e);
                     println!("Error capturing screen. Stopping.");
                     break;
                 }
            };

            // --- Construct Next Request ---
            let acknowledged_safety_checks: Option<Vec<SafetyCheck>> = if computer_call.pending_safety_checks.is_empty() {
                None
            } else {
                warn!("Received pending safety checks: {:?}. Acknowledging all for now.", computer_call.pending_safety_checks);
                Some(computer_call.pending_safety_checks.clone())
            };

            // 1. Construct the ComputerCallOutput enum variant (Screenshot)
            let output_enum_variant = ComputerCallOutput::Screenshot {
                file_id: None,
                image_url: Some(format!("data:image/png;base64,{}", screenshot_base64)),
            };

            // 2. Construct the ComputerToolCallOutput struct
            let output_struct = openai_responses::types::item::ComputerToolCallOutput {
                id: None,
                status: None,
                call_id: call_id.clone(),
                output: output_enum_variant,
                acknowledged_safety_checks,
            };

            // 3. Construct the InputListItem enum variant containing the struct
            // *** Use correct variant InputItem::ComputerToolCallOutput ***
            let next_input_list_item = InputListItem::Item(InputItem::ComputerToolCallOutput(output_struct)); // Wrap struct in Item variant

            // 4. Construct the Input enum variant containing the list item
            let next_input_vec = Input::List(vec![next_input_list_item]); // Use Input::List, wrap in outer Vec

            // 5. Construct the final Request
            current_request = Request {
                model: Model::ComputerUsePreview, // Assuming this variant exists
                input: next_input_vec, // Assign the Vec<Input>
                tools: Some(vec![computer_tool.clone()]),
                previous_response_id: Some(current_response_id),
                truncation: Some(Truncation::Auto),
                ..Default::default()
            };

            // Continue the loop

        } else {
            // No computer_call found, task is finished or model responded with text
            info!("No further computer actions requested.");
            println!("\n--- Final Output ---");
            // Parse final text output from response.output
            for item in response.output {
                 match item {
                     // TODO: Verify actual variant name for text output (e.g., Message, TextData?)
                     OutputItem::Message(msg_item) => { // Guessing variant name
                        // TODO: Verify structure of msg_item and how to get text
                         println!("Message: {:?}", msg_item);
                     }
                     OutputItem::Reasoning(reasoning_item) => {
                         // summary is Vec<ReasoningSummary>, not Option
                         for summary_item in reasoning_item.summary {
                             // TODO: Check actual structure of ReasoningSummary type
                             println!("Reasoning: {:?}", summary_item);
                         }
                     }
                     _ => {}
                 }
            }
            println!("--------------------");
            break; // Exit loop
        }
    } // End loop

    info!("Exiting AI Client.");
    Ok(())
}

// Helper function to call an MCP tool and handle potential errors
// *** Updated signature to take Peer<RoleClient> ***
async fn call_mcp_tool<P: Serialize + std::fmt::Debug>(mcp_peer: &Peer<RoleClient>, tool_name: &str, params: P) -> Result<()> {
    info!("Calling MCP tool '{}' with params: {:?}", tool_name, params);
    let arguments = serde_json::to_value(params).context("Failed to serialize MCP tool parameters")?;
    let arguments_map = match arguments {
        Value::Object(map) => Some(map),
        Value::Null => None,
        _ => return Err(anyhow!("MCP tool parameters did not serialize to a JSON object or null")),
    };
    let mcp_request = CallToolRequestParam { name: tool_name.to_string().into(), arguments: arguments_map, };
    // *** Use mcp_peer.call_tool directly ***
    mcp_peer.call_tool(mcp_request).await.context(format!("MCP tool call '{}' failed", tool_name))?;
    Ok(())
}

// Helper function to call capture_screen and extract base64 data
// *** Updated signature to take Peer<RoleClient> ***
async fn call_capture_screen(
    mcp_peer: &Peer<RoleClient>,
    x: Option<i32>, y: Option<i32>, width: Option<u32>, height: Option<u32>
) -> Result<String> {
    let params = CaptureScreenParams { x, y, width, height };
    // *** Pass mcp_peer directly ***
    let mcp_result = call_mcp_tool_with_result(mcp_peer, "capture_screen", params).await?;
    match mcp_result.content.into_iter().next() {
        Some(content) => match content.raw {
            RawContent::Text(raw_text) => {
                match serde_json::from_str::<ScreenshotResultData>(&raw_text.text) {
                    Ok(data) => data.base64_data.ok_or_else(|| anyhow!("'base64_data' field missing in capture_screen result")),
                    Err(e) => Err(anyhow!("Failed to parse capture_screen JSON result: {}", e)),
                }
            }
            _ => Err(anyhow!("capture_screen returned non-text content")),
        },
        None => Err(anyhow!("capture_screen returned no content")),
    }
}

// Helper to call MCP tool and get Result
// *** Updated signature to take Peer<RoleClient> ***
async fn call_mcp_tool_with_result<P: Serialize + std::fmt::Debug>(
    mcp_peer: &Peer<RoleClient>,
    tool_name: &str,
    params: P
) -> Result<rmcp::model::CallToolResult> {
    info!("Calling MCP tool '{}' with params: {:?}", tool_name, params);
    let arguments = serde_json::to_value(params).context("Failed to serialize MCP tool parameters")?;
    let arguments_map = match arguments {
        Value::Object(map) => Some(map),
        Value::Null => None,
        _ => return Err(anyhow!("MCP tool parameters did not serialize to a JSON object or null")),
    };
    let mcp_request = CallToolRequestParam { name: tool_name.to_string().into(), arguments: arguments_map, };
    // *** Use mcp_peer.call_tool directly ***
    mcp_peer.call_tool(mcp_request).await.context(format!("MCP tool call '{}' failed", tool_name))
}

// Removed add_message_to_history function
