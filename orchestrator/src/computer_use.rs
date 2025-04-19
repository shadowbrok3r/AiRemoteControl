use anyhow::{anyhow, Context, Result};
// *** Using openai_responses SDK ***
use openai_responses::{
    // Main client and request/response types
    types::{
        config::Truncation,
        item::{ComputerAction, ComputerCallOutput, ComputerToolCall, SafetyCheck, ClickButton}, // Added ClickButton
        request::{ContentItem, Input, Request}, // Removed ImageUrl (part of ContentItem)
        response::{Response, OutputItem},
        tools::{Environment, Tool},
        Model, // Removed Role (using string)
    },
    Client as ResponsesClient,
};
// Keep rmcp imports for server interaction
use rmcp::{
    model::{CallToolRequestParam, ErrorCode, ErrorData, RawContent},
    service::{PeerSink, RoleClient, RunningService}, // Added PeerSink back (needed for helpers)
    serve_client,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::{collections::VecDeque, env}; // Removed VecDeque if add_message_to_history is removed
use tokio::net::TcpSocket;
use tracing::{debug, error, info, warn};
// Removed async_openai import

// Configuration
const MCP_SERVER_ADDR: &str = "127.0.0.1:9001";
const COMPUTER_USE_MODEL: &str = "computer-use-preview";
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


#[tokio::main]
async fn main() -> Result<()> {
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
        display_width: DISPLAY_WIDTH as u64, // Cast to u64
        display_height: DISPLAY_HEIGHT as u64, // Cast to u64
        environment: ENVIRONMENT,
        // current_url: None, // Optional
    };

    // Construct the initial request using SDK types
    // TODO: Verify Model enum variant for computer-use-preview
    // Using Gpt4o as a placeholder, assuming it might support it, or use Custom if available
    let initial_request = Request {
        model: Model::ComputerUsePreview, // Placeholder - Verify correct model enum variant
        // model: Model::Custom(COMPUTER_USE_MODEL.to_string()), // If Custom variant exists
        // *** Use Input::UserMessage variant (guessing name) ***
        input: vec![Input::UserMessage { // Assuming Input is an enum
            role: "user".to_string(),
            content: vec![ContentItem::Text { text: user_input.to_string() }],
        }],
        tools: Some(vec![computer_tool.clone()]),
        instructions: None,
        previous_response_id: None,
        reasoning: None,
        truncation: Some(Truncation::Auto),
        ..Default::default()
    };

    // --- Main Computer Use Loop ---
    let mut current_request = initial_request;
    let mut last_response_id: Option<String> = None;

    loop {
        debug!("Sending request...");
        // *** Use ? to handle Result<Response, Error> ***
        let response: Response = openai_client.create(current_request.clone()).await
            .context("OpenAI Responses API call failed")?; // Use context before ?
        // info!("Received response: {:?}", response);

        last_response_id = response.id.clone();

        // Find the computer_call action in the response output
        // *** Use OutputItem::ComputerToolCall variant ***
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
                    // *** Match on button enum, don't use .to_string() ***
                    let button_str = match button {
                         ClickButton::Left => "left",
                         ClickButton::Right => "right",
                         ClickButton::Middle => "middle",
                         // Add other variants if present in SDK's ClickButton enum
                         // _ => return Err(anyhow!("Unsupported ClickButton variant received from SDK"))
                    }.to_string();
                    // *** Remove dereference from x, y (assuming they are u64/f64) ***
                    let params = OpenAIClickParams { x: *x as i32, y: *y as i32, button: button_str }; // Cast u64/f64 to i32 if needed
                    call_mcp_tool(&mcp_peer, "execute_openai_click", params).await
                }
                ComputerAction::Scroll { x, y, scroll_x, scroll_y } => {
                     // *** Remove dereference from x, y, scroll_x, scroll_y ***
                    let params = OpenAIScrollParams { x: *x as i32, y: *y as i32, scroll_x: *scroll_x as i32, scroll_y: *scroll_y as i32 }; // Cast
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
                 // *** Match Wait variant payload correctly ***
                 ComputerAction::Wait { duration } => { // Assuming field name is duration
                    let params = OpenAIWaitParams { duration_ms: Some(*duration) }; // Use Option<u64>
                    call_mcp_tool(&mcp_peer, "execute_openai_wait", params).await
                 }
                _ => {
                     warn!("Received unknown or unhandled action type: {:?}", action);
                     Err(anyhow!("Unknown or unhandled action type received: {:?}", action))
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

            // Construct the output Input item using SDK types
            // TODO: Verify the exact structure for ComputerCallOutput and its nested 'output' field
            let output_content = ContentItem::ImageUrl { // Guessing ContentItem::ImageUrl is used here
                 image_url: openai_responses::types::request::ImageUrl { // Use SDK's ImageUrl struct
                    url: format!("data:image/png;base64,{}", screenshot_base64),
                    detail: None, // Detail might not apply here, or use ImageDetail::Auto
                 }
            };

            // TODO: Verify the exact structure for ComputerCallOutput's 'output' field.
            // The user-provided struct had `output: ComputerCallOutput`, which seemed recursive.
            // The OpenAI JSON shows `output: { type: "input_image", image_url: "..." }`
            // Let's assume the SDK expects a `Value` or specific struct here. Using Value for now.
            let output_payload: Value = serde_json::to_value(output_content)
                .context("Failed to serialize screenshot ContentItem")?;

            let output_item = ComputerCallOutput {
                call_id: call_id.clone(),
                acknowledged_safety_checks,
                output: output_payload, // Assign the Value, verify SDK type!
            };

            // Prepare the next request using Input::ComputerCallOutput variant (guessing name)
            // TODO: Verify the correct Input enum variant for sending tool output
             let next_input = Input::ComputerCallOutput { // Assuming Input is an enum with this variant
                 call_id: call_id, // Pass call_id here
                 output: output_item, // Pass the constructed output item
                 // Remove redundant/incorrect fields based on enum variant definition
                 // r#type: "computer_call_output".to_string(),
                 // role: "".to_string(),
                 // content: vec![],
                 // acknowledged_safety_checks: None,
             };

            current_request = Request {
                // TODO: Verify Model enum variant for computer-use-preview
                model: Model::Gpt4o, // Placeholder
                input: vec![next_input],
                tools: Some(vec![computer_tool.clone()]),
                previous_response_id: last_response_id.clone(),
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
                     // *** Check actual variant name for text output ***
                     OutputItem::Text(text_item) => { // Assuming OutputItem::Text variant exists
                         println!("{}", text_item.text.unwrap_or_default()); // Assuming text field exists
                     }
                      // *** Check actual variant name for reasoning ***
                     OutputItem::Reasoning(reasoning_item) => {
                         // *** Handle Option for summary ***
                         if let Some(summary_vec) = reasoning_item.summary {
                             for summary_item in summary_vec {
                                 // TODO: Check actual structure of ReasoningSummary type
                                 println!("Reasoning: {:?}", summary_item);
                             }
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
async fn call_mcp_tool<P: Serialize + std::fmt::Debug>(mcp_peer: &PeerSink, tool_name: &str, params: P) -> Result<()> {
    info!("Calling MCP tool '{}' with params: {:?}", tool_name, params);
    let arguments = serde_json::to_value(params).context("Failed to serialize MCP tool parameters")?;
    let arguments_map = match arguments {
        Value::Object(map) => Some(map),
        Value::Null => None,
        _ => return Err(anyhow!("MCP tool parameters did not serialize to a JSON object or null")),
    };
    let mcp_request = CallToolRequestParam { name: tool_name.to_string().into(), arguments: arguments_map, };
    mcp_peer.call_tool(mcp_request).await.context(format!("MCP tool call '{}' failed", tool_name))?;
    Ok(())
}

// Helper function to call capture_screen and extract base64 data
async fn call_capture_screen(
    mcp_peer: &PeerSink,
    x: Option<i32>, y: Option<i32>, width: Option<u32>, height: Option<u32>
) -> Result<String> {
    let params = CaptureScreenParams { x, y, width, height };
    let mcp_result = call_mcp_tool_with_result(mcp_peer, "capture_screen", params).await?;
    match mcp_result.content.into_iter().next() { // *** Use .content ***
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
async fn call_mcp_tool_with_result<P: Serialize + std::fmt::Debug>(
    mcp_peer: &PeerSink,
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
    mcp_peer.call_tool(mcp_request).await.context(format!("MCP tool call '{}' failed", tool_name))
}

// Removed add_message_to_history function and MAX_CONVERSATION_DEPTH constant
