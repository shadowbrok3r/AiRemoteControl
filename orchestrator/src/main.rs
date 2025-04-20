use anyhow::{Context, Result};
use async_openai::{
    config::Config,
    error::OpenAIError,
    types::{
        ChatCompletionRequestAssistantMessage, ChatCompletionRequestAssistantMessageContent,
        ChatCompletionRequestMessage, ChatCompletionRequestMessageContentPartImage,
        ChatCompletionRequestMessageContentPartText, ChatCompletionRequestSystemMessage, ChatCompletionRequestSystemMessageContent,
        ChatCompletionRequestToolMessage, ChatCompletionRequestToolMessageContent, ChatCompletionRequestUserMessage,
        ChatCompletionRequestUserMessageContent, ChatCompletionTool, ChatCompletionToolChoiceOption,
        ChatCompletionToolType, CreateChatCompletionRequest, FunctionObject, ImageDetail, ImageUrl
    },
    Client as OpenAIClient,
};
use rmcp::{
    model::{CallToolRequestParam, RawContent}, 
    service::{RoleClient, RunningService},
    serve_client,
};
use serde_json::{json, Map, Value};
use std::{collections::VecDeque, env};
use tokio::net::TcpSocket; 
use tracing::{info, error, debug, warn};
use futures::stream::StreamExt;
use futures::future::join_all;
use tokio::task::JoinHandle;
use std::collections::HashMap;

pub mod computer_use;

// Configuration
const MCP_SERVER_ADDR: &str = "127.0.0.1:9001"; // Address of your TCP MCP Server
const MAX_CONVERSATION_DEPTH: usize = 15; // Max history items (including System prompt)
const OPENAI_CHAT_MODEL: &str = "gpt-4.1-mini"; // Or your preferred model like gpt-4o-mini if desired
const OPENAI_VISION_MODEL: &str = "gpt-4.1-nano"; // Specific model for image analysis
// const OPENAI_CHAT_MODEL: &str = "gemini-2.0-flash"; // Or your preferred model like gpt-4o-mini if desired
// const OPENAI_VISION_MODEL: &str = "gemini-2.0-flash"; // Specific model for image analysis

#[derive(Debug, Clone, Default)]
struct PartialToolCall {
    index: Option<usize>,
    id: Option<String>,
    name: Option<String>,
    /// Accumulate argument chunks
    arguments: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    // i have to wait for the computer-use-model to become available. it is only allowed for 'select' devs
    // computer_use::run_computer_use().await?;
    run_gpt_computer_use().await?;
    info!("Exiting AI Client.");
    Ok(())
}

async fn run_gpt_computer_use() -> anyhow::Result<(), anyhow::Error> {

    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env().add_directive(tracing::Level::INFO.into()))
        .with_writer(std::io::stdout)
        .with_ansi(true)
        .init();

    // Load OpenAI API Key
    dotenv::dotenv().ok();
    if env::var("OPENAI_API_KEY").is_err() {
        anyhow::bail!("OPENAI_API_KEY environment variable not set.");
    } 

    // let gemini_key = dotenv::env::var("GEMINI_KEY").unwrap();

    // let c = OpenAIConfig::new()
    //     .with_api_base("https://generativelanguage.googleapis.com/v1beta/openai/")
    //     .with_api_key("");

    // let openai_client = OpenAIClient::with_config(c);
    let openai_client = OpenAIClient::new();

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

    // --- Fetch Tools from MCP Server ---
    info!("Fetching tools from MCP server...");
    let mcp_tools_result = mcp_peer
        .list_tools(None) // Use None for default options
        .await
        .context("Failed to list tools from MCP server")?;
    info!("Available tools: {:#?}", mcp_tools_result.tools.iter().map(|t| &t.name).collect::<Vec<_>>());

    // Convert MCP tools to OpenAI tool format
    let openai_tools: Vec<ChatCompletionTool> = mcp_tools_result
        .tools
        .into_iter()
        .map(|mcp_tool| {
            // Schema Patching Logic
            let parameters_value: Option<Value> = {
                let needs_patch = mcp_tool.input_schema.get("properties").is_none();
                if needs_patch {
                    info!("Patching schema for parameterless tool: {}", mcp_tool.name);
                    Some(json!({"type": "object", "properties": {}}))
                } else {
                    Some(Value::Object(mcp_tool.input_schema.as_ref().clone()))
                }
            };


            ChatCompletionTool {
                r#type: ChatCompletionToolType::Function,
                function: FunctionObject {
                    name: mcp_tool.name.to_string(),
                    description: Some(mcp_tool.description.to_string()),
                    parameters: parameters_value,
                    strict: None,
                },
            }
        })
        .collect();

    if openai_tools.is_empty() {
         warn!("No tools with schemas found on the server. OpenAI cannot use tools.");
    } else {
         info!("{} tools converted for OpenAI.", openai_tools.len());
    }


    // --- Main Interaction Loop ---
    let mut conversation_history: VecDeque<ChatCompletionRequestMessage> = VecDeque::new();
    let system_prompt = r#"You are a helpful AI assistant designed to control the user's desktop via function calls.

    **Core Functionality:**
    * Analyze user requests carefully.
    * Break down complex tasks (like finding a window, typing in it, and then moving it) into a sequence of individual tool calls.
    * Use the available tools step-by-step to fulfill the request.
    * Execute tools sequentially unless the user explicitly asks for parallel actions *and* the actions are independent.

    **Tool Usage Guidelines:**
    * **`find_window`**: Use this first to locate a window by its title before interacting with it. Note the returned coordinates (x, y) and dimensions.
    * **`move_mouse`**: Moves the cursor to absolute or relative coordinates.
    * **`mouse_action`**: Performs clicks, presses, or releases.
        * **Click:** `button: "Left", click_type: "Click"` (or omit `click_type`).
        * **Press & Hold:** `button: "Left", click_type: "Press"`.
        * **Release:** `button: "Left", click_type: "Release"`.
    * **`keyboard_action`**: Types text or simulates key presses (like Enter, Ctrl+C).
    * **`run_shell_command`**: Executes commands like opening applications (e.g., `command: "notepad"`).
    * **`capture_screen`**: Captures the screen. Use the resulting text description (which includes vision model analysis) for subsequent analysis or actions. Do not attempt to interpret the base64 data directly.

    **Complex Actions (Example: Dragging a Window):**
    1.  Use `find_window` to get the window's position (e.g., title bar coordinates `x`, `y`).
    2.  Call `move_mouse` to position the cursor on the title bar (e.g., `x`, `y + 10`).
    3.  Call `wait(duration_ms=150)` to ensure the cursor is settled.
    4.  Call `mouse_action` with `button: "Left", click_type: "Press"` to grab the title bar.
    5.  Call `wait(duration_ms=100)` to ensure the press is registered.
    6.  Call `move_mouse` to the *new* desired window position (e.g., `new_x`, `new_y + 10`).
    7.  Call `wait(duration_ms=100)` to ensure the move is complete.
    8.  Call `mouse_action` with `button: "Left", click_type: "Release"` to drop the window.

    **Interaction:**
    * Ask for clarification if a request is ambiguous or requires information you don't have (e.g., "Where should I move the window?").
    * Inform the user upon successful completion of the overall task.
    * Report any errors encountered during tool execution."#.to_string();


    // Add initial system message
    conversation_history.push_back(ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage{
        content: ChatCompletionRequestSystemMessageContent::Text(system_prompt.clone()), 
        name: None
    }));


    loop { // Outer loop (user input)
        // Get user input
        println!("\nEnter your request (or type 'quit'):");
        let mut user_input = String::new();
        std::io::stdin().read_line(&mut user_input)?;
        let user_input = user_input.trim();

        if user_input.eq_ignore_ascii_case("quit") {
            return Ok(());
        }
        if user_input.is_empty() {
            continue;
        }

        // Add User message directly
        conversation_history.push_back(ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessage {
                content: ChatCompletionRequestUserMessageContent::Text(user_input.to_string()),
                name: None,
            }
        ));

        // --- Call OpenAI Loop (Handles potential multi-step tool calls) ---
        loop { // Inner loop (OpenAI calls)

            // --- Trim History ---
            // Keep the system prompt (index 0) and trim older messages from index 1 if history exceeds max depth
            while conversation_history.len() > MAX_CONVERSATION_DEPTH {
                if conversation_history.len() >= 2 { // Ensure System prompt + one other exists
                    info!("Trimming history: Removing message at index 1. Current length: {}", conversation_history.len());
                    // *** Add logging to see what's being removed ***
                    let removed_message_role = match conversation_history.get(1) {
                        Some(ChatCompletionRequestMessage::User(_)) => "User",
                        Some(ChatCompletionRequestMessage::Assistant(_)) => "Assistant",
                        Some(ChatCompletionRequestMessage::Tool(_)) => "Tool",
                        Some(ChatCompletionRequestMessage::System(_)) => "System (Error!)", // Should not happen
                        Some(ChatCompletionRequestMessage::Function(_)) => "Function (Error!)", // Should not happen
                        Some(ChatCompletionRequestMessage::Developer(_)) => "Developer (Error!)", // Should not happen
                        None => "None (Error!)",
                   };

                   info!(
                        "Trimming history: Removing message at index 1 (Role: {}). Current length: {}",
                        removed_message_role,
                        conversation_history.len()
                    );
                    conversation_history.remove(1); // Remove oldest non-system message
                } else {
                    warn!("Attempted to trim history below 2 messages. Breaking trim loop.");
                    break; // Should not happen if MAX_DEPTH >= 1
                }
            }
            // --- End Trim History ---

            // *** Defensive Check ***
            if conversation_history.len() >= 2 {
                if let Some(ChatCompletionRequestMessage::Tool(_)) = conversation_history.get(1) {
                        // This should NOT happen with the current logic if VecDeque::remove(1) works as expected.
                        error!("CRITICAL: History state invalid after trimming! Message at index 1 is Tool.");
                        debug!("Invalid History State: {:#?}", conversation_history);
                        // Handle this critical error, maybe break or return?
                        println!("Internal error: Invalid conversation history state detected. Please report this.");
                        break; // Break inner loop
                }
            }

            info!("Sending request to OpenAI chat model...");
            info!("Conversation History (len={}): {:#?}", conversation_history.len(), conversation_history); // Log length and content

            let request = CreateChatCompletionRequest {
                model: OPENAI_CHAT_MODEL.to_string(),
                messages: conversation_history.iter().cloned().collect(), // Use current trimmed history
                tools: if openai_tools.is_empty() { None } else { Some(openai_tools.clone()) },
                tool_choice: if openai_tools.is_empty() { None } else { Some(ChatCompletionToolChoiceOption::Auto) },
                stream: Some(true),
                parallel_tool_calls: Some(true),
                ..Default::default()
            };

            // --- Stream Handling ---
            let mut stream = match openai_client.chat().create_stream(request).await {
                Ok(s) => s,
                Err(e) => {
                    error!("OpenAI API stream creation error: {}", e);
                    // Handle error appropriately (e.g., print message, break inner loop)
                     match e {
                         OpenAIError::ApiError(api_error) => { error!("--> API Error Details: {:#?}", api_error); }
                         OpenAIError::Reqwest(req_err) => { error!("--> Network Error Details: {:#?}", req_err); }
                         _ => { error!("--> Other OpenAI Error: {:#?}", e); }
                     }
                     println!("Error starting communication with OpenAI. Check logs.");
                     break; // Break inner loop
                }
            };

            let mut full_response_content = String::new();
            // Use HashMap to reconstruct tool calls based on index from deltas
            let mut partial_tool_calls: HashMap<u32, PartialToolCall> = HashMap::new();
            let mut final_tool_calls: Vec<async_openai::types::ChatCompletionMessageToolCall> = Vec::new(); // Store fully formed calls

            print!("\nAssistant (Streaming): "); // Indicate streaming start
            while let Some(result) = stream.next().await {
                match result {
                    Ok(stream_response) => {
                        for choice in stream_response.choices {
                            let delta = choice.delta;

                            // Accumulate content
                            if let Some(content_chunk) = delta.content {
                                print!("{}", content_chunk); // Print content chunk immediately
                                use std::io::Write; // Import Write trait for flush
                                std::io::stdout().flush().unwrap_or_default(); // Ensure chunk is displayed
                                full_response_content.push_str(&content_chunk);
                            }

                            // Accumulate tool calls (handle partial deltas)
                            if let Some(delta_tool_calls) = delta.tool_calls {
                                for tool_call_chunk in delta_tool_calls {
                                    let index = tool_call_chunk.index; // Index is key for reconstruction
                                    let partial = partial_tool_calls.entry(index).or_default();
                                    partial.index = Some(index as usize); // Store index

                                    if let Some(id) = tool_call_chunk.id {
                                        partial.id = Some(id);
                                    }
                                    if let Some(function) = tool_call_chunk.function {
                                        if let Some(name) = function.name {
                                            partial.name = Some(name);
                                        }
                                        if let Some(args_chunk) = function.arguments {
                                            partial.arguments.push_str(&args_chunk);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        error!("Error receiving stream chunk: {}", e);
                        // Handle stream error (e.g., maybe break or try to continue)
                        println!("\nError during streaming response from OpenAI.");
                        // Potentially break or set an error flag
                    }
                }
            }
            println!(); // Newline after streaming finishes


                        // --- Process Accumulated Response ---

            // Finalize reconstructed tool calls
            for (_index, partial) in partial_tool_calls.into_iter() {
                 if let (Some(id), Some(name)) = (partial.id.clone(), partial.name.clone()) {
                     final_tool_calls.push(async_openai::types::ChatCompletionMessageToolCall {
                         id,
                         r#type: async_openai::types::ChatCompletionToolType::Function, // Assuming function type
                         function: async_openai::types::FunctionCall {
                             name,
                             arguments: partial.arguments,
                         },
                     });
                 } else {
                     warn!("Incomplete tool call received via stream delta: index={:?}, partial={:?}", partial.index, partial);
                 }
            }



            // Add the complete Assistant message to history
            let assistant_message = ChatCompletionRequestAssistantMessage {
                content: if full_response_content.is_empty() { None } else { Some(ChatCompletionRequestAssistantMessageContent::Text(full_response_content.clone())) },
                tool_calls: if final_tool_calls.is_empty() { None } else { Some(final_tool_calls.clone()) },
                ..Default::default() // Use default for other fields like name, refusal, audio
            };
            conversation_history.push_back(ChatCompletionRequestMessage::Assistant(assistant_message));

            // --- Handle Tool Calls (Parallel Execution) ---
            if !final_tool_calls.is_empty() {
                info!("Executing {} tool call(s) in parallel...", final_tool_calls.len());

                let mut tool_tasks: Vec<JoinHandle<(String, String, Result<rmcp::model::CallToolResult, rmcp::ServiceError>)>> = Vec::new();

                for tool_call in final_tool_calls {
                    let call_id = tool_call.id.clone();
                    let function_call = tool_call.function;
                    let tool_name = function_call.name;
                    let arguments_str = function_call.arguments;

                    // Parse arguments (handle potential errors)
                    let arguments_map: Option<Map<String, Value>> = match serde_json::from_str(&arguments_str) {
                        Ok(Value::Object(map)) => Some(map),
                        Ok(_) => {
                            warn!("Tool arguments were not a JSON object for tool '{}'. Treating as empty.", tool_name);
                            None
                        }
                        Err(e) => {
                            error!("Failed to parse arguments JSON string for tool '{}': {} (Args: '{}')", tool_name, e, arguments_str);
                            // Immediately add error result for this tool call
                            let error_msg = format!("Invalid JSON arguments provided by AI for tool '{}': {}", tool_name, e);
                            conversation_history.push_back(ChatCompletionRequestMessage::Tool(ChatCompletionRequestToolMessage{
                                tool_call_id: call_id.clone(), // Clone id here
                                content: ChatCompletionRequestToolMessageContent::Text(error_msg)
                            }));
                            continue; // Skip spawning task for this invalid call
                        }
                    };

                    info!("Spawning task for MCP tool '{}' (call_id: {}) with args: {:#?}", tool_name, call_id, arguments_map);

                    let mcp_peer_clone = mcp_peer.clone();
                    let mcp_request = CallToolRequestParam { name: tool_name.clone().into(), arguments: arguments_map };
                    let call_id_clone = call_id.clone();
                    let tool_name_clone = tool_name.clone(); // Clone tool_name for the task

                    // Spawn the MCP tool call task
                    tool_tasks.push(tokio::spawn(async move {
                        let result = mcp_peer_clone.call_tool(mcp_request).await;
                        (call_id_clone, tool_name_clone, result) // Return call_id, tool_name, result
                    }));
                }

                // Wait for all tool call tasks to complete
                let task_results = join_all(tool_tasks).await;
                // Use a temporary vec to store results before adding to history to avoid borrowing issues
                let mut tool_message_results = Vec::new();

                // Process results and add Tool messages to history
                for task_result in task_results {
                    match task_result {
                        Ok((call_id, tool_name, mcp_call_result)) => {
                            // Process the result in an async block to allow calling analyze_image_with_vision
                            let tool_result_content_str = async {
                                match mcp_call_result {
                                    Ok(mcp_result_data) => {
                                        info!("MCP tool '{}' (call_id: '{}') executed successfully.", tool_name, call_id);
                                        match mcp_result_data.content.into_iter().next() {
                                            Some(content) => match content.raw {
                                                RawContent::Text(raw_text) => {
                                                    // <<< Check if it was capture_screen >>>
                                                    if tool_name == "capture_screen" {
                                                        info!("Processing capture_screen result (call_id: {})...", call_id);
                                                        match serde_json::from_str::<Value>(&raw_text.text) {
                                                            Ok(json_val) => {
                                                                if let Some(base64_data) = json_val.get("base64_data").and_then(|v| v.as_str()) {
                                                                    let vision_prompt = "Describe this screenshot in detail, focusing on visible text, UI elements, and overall layout.".to_string();
                                                                    // Call vision analysis
                                                                    match analyze_image_with_vision(&openai_client, vision_prompt, base64_data).await {
                                                                        Ok(desc) => { info!("Vision analysis successful for call_id: {}", call_id); desc }
                                                                        Err(e) => { error!("Vision analysis failed for call_id '{}': {}", call_id, e); format!("Screenshot captured but vision analysis failed: {}", e) }
                                                                    }
                                                                } else {
                                                                    warn!("capture_screen JSON missing 'base64_data' for call_id: {}", call_id);
                                                                    raw_text.text // Return raw JSON if no base64
                                                                }
                                                            }
                                                            Err(e) => {
                                                                warn!("Failed to parse capture_screen JSON for call_id '{}': {}. Returning raw text.", call_id, e);
                                                                raw_text.text // Return raw text if parse fails
                                                            }
                                                        }
                                                    } else {
                                                        raw_text.text // Return text for other tools
                                                    }
                                                }
                                                _ => format!("Tool '{}' (call_id: '{}') returned non-text content.", tool_name, call_id),
                                            },
                                            None => format!("Tool '{}' (call_id: '{}') returned no content.", tool_name, call_id),
                                        }
                                    }
                                    Err(e) => {
                                        error!("MCP tool '{}' (call_id: '{}') failed: {}", tool_name, call_id, e);
                                        json!({ "status": "error", "message": format!("Failed MCP execution for tool '{}' (call_id: '{}'): {}", tool_name, call_id, e) }).to_string()
                                    }
                                }
                            }.await; // Await the async block processing the result

                            // Store the result to be added later
                            tool_message_results.push(ChatCompletionRequestMessage::Tool(ChatCompletionRequestToolMessage{
                                tool_call_id: call_id,
                                content: ChatCompletionRequestToolMessageContent::Text(tool_result_content_str)
                            }));
                        }
                        Err(join_err) => {
                            error!("Tool execution task failed to join: {}", join_err);
                            // Optionally add a generic error message to history here if needed
                        }
                    }
                }

                // *** Add the collected tool results to the main history ***
                info!("Adding {} tool result messages to history.", tool_message_results.len());
                for msg in tool_message_results {
                    conversation_history.push_back(msg);
                }

                // After processing all tool results, continue the inner loop to send them back
                continue;

            } else if !full_response_content.is_empty() {
                // --- Handle Regular Assistant Message (No Tool Calls) ---
                info!("OpenAI final streaming response: {}", full_response_content);
                // Already printed during streaming, just break
                break; // Break inner loop, wait for next user input

            } else {
                 warn!("OpenAI stream finished with no content and no tool calls.");
                 println!("Assistant provided an empty response.");
                 break; // Break inner loop
            }
        } // End inner OpenAI loop

    } // End main user input loop
}

// Vision analysis function (remains the same)
async fn analyze_image_with_vision<C: Config>(
    client: &OpenAIClient<C>, // Use Client<C>
    prompt: String,
    base64_image: &str,
) -> Result<String> {
    info!("Calling vision model '{}'...", OPENAI_VISION_MODEL);

    let data_url = format!("data:image/png;base64,{}", base64_image);

    // Create the request message with text and image parts
    let request_message = ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
        content: ChatCompletionRequestUserMessageContent::Array(vec![
            async_openai::types::ChatCompletionRequestUserMessageContentPart::Text(ChatCompletionRequestMessageContentPartText {
                text: prompt,
            }),
            async_openai::types::ChatCompletionRequestUserMessageContentPart::ImageUrl(ChatCompletionRequestMessageContentPartImage {
                image_url: ImageUrl {
                    url: data_url,
                    detail: Some(ImageDetail::Auto), // Or High / Low
                },
            }),
        ]),
        name: None,
    });

    // Create the chat completion request for the vision model
    let request = CreateChatCompletionRequest {
        model: OPENAI_VISION_MODEL.to_string(),
        messages: vec![request_message],
        ..Default::default()
    };

    // Call the API and log detailed errors
    let response = client.chat().create(request).await
        // *** Added detailed error logging using map_err ***
        .map_err(|e| {
            error!("Vision API call failed. Error details: {:#?}", e);
             match &e {
                 OpenAIError::ApiError(api_error) => {
                      error!("--> Specific Vision API Error: {:#?}", api_error);
                 }
                 OpenAIError::Reqwest(req_err) => {
                       error!("--> Specific Vision Network Error: {:#?}", req_err);
                 }
                 _ => {} // Other error types already logged by the top-level error! macro
             }
            anyhow::anyhow!(e).context("Vision API call failed") // Convert to anyhow::Error
        })?;

    // Extract the text response
    if let Some(choice) = response.choices.into_iter().next() {
        if let Some(content) = choice.message.content {
            Ok(content)
        } else {
            Ok("Vision model returned no text content.".to_string())
        }
    } else {
        Ok("Vision model returned no choices.".to_string())
    }
}

// Removed the add_message_to_history function
