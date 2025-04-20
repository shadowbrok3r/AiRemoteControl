use anyhow::{Context, Result}; // Added anyhow for bail macro
use async_openai::{
    config::Config, // *** Added Config import ***
    error::OpenAIError, // *** Added OpenAIError import ***
    types::{
        ChatCompletionRequestAssistantMessage, ChatCompletionRequestAssistantMessageContent,
        ChatCompletionRequestAssistantMessageContentPart, ChatCompletionRequestMessage, ChatCompletionRequestMessageContentPartImage,
        ChatCompletionRequestMessageContentPartText, ChatCompletionRequestSystemMessage, ChatCompletionRequestSystemMessageContent,
        ChatCompletionRequestToolMessage, ChatCompletionRequestToolMessageContent, ChatCompletionRequestUserMessage,
        ChatCompletionRequestUserMessageContent, ChatCompletionTool, ChatCompletionToolChoiceOption,
        ChatCompletionToolType, CreateChatCompletionRequest, FunctionObject, ImageDetail, ImageUrl, Role // *** Changed ImageUrlDetail to ImageDetail ***
    },
    Client as OpenAIClient, // Alias for Client<C>
};
use rmcp::{
    // Added ListToolsRequest, RawContent. Removed Content, Tool alias.
    model::{CallToolRequestParam, RawContent}, // Removed ListToolsRequest import error, use None below
    service::{RoleClient, RunningService}, // RoleClient likely needs feature flag
    serve_client, // serve_client likely needs feature flag
};
use serde_json::{json, Map, Value}; // Added Map
use std::{collections::VecDeque, env};
use tokio::net::TcpSocket; // Use TcpSocket for connecting
use tracing::{info, error, debug, warn};

pub mod computer_use;

// Configuration
const MCP_SERVER_ADDR: &str = "127.0.0.1:9001"; // Address of your TCP MCP Server
const MAX_CONVERSATION_DEPTH: usize = 15; // Max history items (including System prompt)
const OPENAI_CHAT_MODEL: &str = "gpt-4.1-mini"; // Or your preferred model like gpt-4o-mini if desired
// *** Added Vision Model constant ***
const OPENAI_VISION_MODEL: &str = "gpt-4.1-nano"; // Specific model for image analysis

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
    info!("Available tools: {:?}", mcp_tools_result.tools.iter().map(|t| &t.name).collect::<Vec<_>>());

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
    let system_prompt = "You are a helpful AI assistant. You can control the user's desktop using the available tools. Analyze the user's request and use the tools provided to fulfill it step-by-step. Ask for clarification if the request is ambiguous. Inform the user upon successful completion or if an error occurs. When asked to analyze a screen capture, use the textual description provided as the result of the 'capture_screen' tool.".to_string();

    // Add initial system message
    conversation_history.push_back(ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage{
        content: ChatCompletionRequestSystemMessageContent::Text(system_prompt.clone()), // Clone system prompt text
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
                        debug!("Invalid History State: {:?}", conversation_history);
                        // Handle this critical error, maybe break or return?
                        println!("Internal error: Invalid conversation history state detected. Please report this.");
                        break; // Break inner loop
                }
            }

            info!("Sending request to OpenAI chat model...");
            info!("Conversation History (len={}): {:?}", conversation_history.len(), conversation_history); // Log length and content

            let request = CreateChatCompletionRequest {
                model: OPENAI_CHAT_MODEL.to_string(),
                messages: conversation_history.iter().cloned().collect(), // Use current trimmed history
                tools: if openai_tools.is_empty() { None } else { Some(openai_tools.clone()) },
                tool_choice: if openai_tools.is_empty() { None } else { Some(ChatCompletionToolChoiceOption::Auto) },
                ..Default::default()
            };

            let response = match openai_client.chat().create(request).await {
                 Ok(res) => res,
                 Err(e) => {
                     // *** Use match on OpenAIError variants instead of downcast_ref ***
                     error!("OpenAI API error: {}", e);
                     match e {
                         OpenAIError::ApiError(api_error) => {
                              error!("OpenAI API Error Details: {:?}", api_error);
                              if let Some(code) = &api_error.code {
                                   println!("OpenAI Error Code: {}", code);
                              }
                         }
                         OpenAIError::Reqwest(req_err) => {
                              error!("Network Error Details: {:?}", req_err);
                         }
                         // Add other OpenAIError variants as needed
                         _ => {
                              error!("Other OpenAI Error: {:?}", e);
                         }
                     }
                     println!("Error communicating with OpenAI. Please check logs and API key/quota. Try again.");
                     // Don't pop the user message if the API call fails, allow user to retry or quit
                     // conversation_history.pop_back(); // Removed this line
                     break; // Break inner loop, go back to waiting for user input
                 }
            };


            let choice = match response.choices.into_iter().next() {
                 Some(ch) => ch,
                 None => {
                     warn!("No choices received from OpenAI.");
                     println!("OpenAI did not provide a response.");
                     break; // Break inner loop
                 }
            };

            // Construct the assistant message based on the response choice
            let assistant_message_request = ChatCompletionRequestAssistantMessage {
                content: choice.message.content.clone().map(ChatCompletionRequestAssistantMessageContent::Text),
                name: None,
                tool_calls: choice.message.tool_calls.clone(),
                function_call: choice.message.function_call.clone(), // Keep for older models if needed
                audio: None, // Assuming no audio response for now
                refusal: None, // Assuming no refusal content for now
            };

            // Add Assistant message directly
            conversation_history.push_back(ChatCompletionRequestMessage::Assistant(assistant_message_request.clone()));


            // --- Handle Tool Calls ---
            if let Some(tool_calls) = assistant_message_request.tool_calls {
                info!("OpenAI requested {} tool call(s).", tool_calls.len());

                for tool_call in tool_calls {
                    let call_id = tool_call.id.clone();
                    let function_call = tool_call.function;
                    let tool_name = function_call.name;
                    let arguments_str = function_call.arguments;

                    let arguments_value: Value = match serde_json::from_str(&arguments_str) {
                        Ok(args) => args,
                        Err(e) => {
                            error!("Failed to parse arguments JSON string for tool '{}': {} (Args: '{}')", tool_name, e, arguments_str);
                            let error_msg = format!("Invalid JSON arguments provided by AI for tool '{}': {}", tool_name, e);
                            // Add Tool message with error directly
                            conversation_history.push_back(ChatCompletionRequestMessage::Tool(ChatCompletionRequestToolMessage{
                                tool_call_id: call_id,
                                content: ChatCompletionRequestToolMessageContent::Text(error_msg)
                            }));
                            continue; // Skip this tool call, proceed to next tool call or next API iteration
                        }
                    };

                    // Ensure arguments are a JSON object for MCP
                    let arguments_map: Option<Map<String, Value>> = match arguments_value {
                        Value::Object(map) => Some(map),
                        _ => {
                            warn!("Tool arguments were not a JSON object for tool '{}'. Treating as empty.", tool_name);
                            None // Send None if not an object
                        }
                    };

                    info!("Calling MCP tool '{}' with args: {:?}", tool_name, arguments_map);

                    // --- Call MCP Server Tool ---
                    let mcp_request = CallToolRequestParam {
                        name: tool_name.clone().into(),
                        arguments: arguments_map,
                    };

                    let tool_result_content_str: String = match mcp_peer.call_tool(mcp_request.clone()).await {
                        Ok(mcp_result) => {
                            info!("MCP tool '{}' executed successfully.", mcp_request.name);
                            // *** Use mcp_result.content, not .contents ***
                            match mcp_result.content.into_iter().next() {
                                Some(content) => {
                                    match content.raw {
                                        RawContent::Text(raw_text_content) => {
                                            // Special handling for screen capture analysis
                                            if mcp_request.name == "capture_screen" {
                                                info!("Received screen capture result, attempting analysis...");
                                                match serde_json::from_str::<Value>(&raw_text_content.text) {
                                                    Ok(json_val) => {
                                                        if let Some(base64_data) = json_val.get("base64_data").and_then(|v| v.as_str()) {
                                                            let vision_prompt = "Describe this screenshot in detail. If there is a text editor open, please also read the text visible within it.".to_string();
                                                            match analyze_image_with_vision(&openai_client, vision_prompt, base64_data).await {
                                                                Ok(description) => {
                                                                    info!("Vision analysis successful.");
                                                                    description
                                                                }
                                                                Err(e) => {
                                                                    error!("Vision analysis failed: {}", e);
                                                                    format!("Screenshot captured but vision analysis failed: {}", e)
                                                                }
                                                            }
                                                        } else {
                                                            warn!("capture_screen result JSON did not contain 'base64_data' string.");
                                                            raw_text_content.text // Return raw JSON if no base64 data
                                                        }
                                                    }
                                                    Err(e) => {
                                                        warn!("Failed to parse capture_screen result as JSON: {}. Returning raw text.", e);
                                                        raw_text_content.text // Return raw text if JSON parsing fails
                                                    }
                                                }
                                            } else {
                                                raw_text_content.text // Return text for other tools
                                            }
                                        }
                                        _ => format!("Tool '{}' returned non-text content.", mcp_request.name),
                                    }
                                }
                                None => format!("Tool '{}' executed successfully but returned no content.", mcp_request.name),
                            }
                        }
                        Err(e) => {
                            error!("Failed to call MCP tool '{}': {}", mcp_request.name, e);
                            // Return error as JSON string for OpenAI
                            json!({
                                "status": "error",
                                "message": format!("Failed to execute tool '{}' via MCP: {}", mcp_request.name, e)
                            }).to_string()
                        }
                    };


                    // Add Tool message with result/error directly
                    conversation_history.push_back(ChatCompletionRequestMessage::Tool(ChatCompletionRequestToolMessage{
                        tool_call_id: call_id,
                        content: ChatCompletionRequestToolMessageContent::Text(tool_result_content_str)
                    }));
                }
                continue; // Continue the inner OpenAI loop to send tool results back

            } else if let Some(content) = assistant_message_request.content {
                // --- Handle Regular Assistant Message ---
                match content {
                    ChatCompletionRequestAssistantMessageContent::Text(text)=>{
                        info!("OpenAI final response: {}",text);
                        println!("\nAssistant:\n{}",text);
                    }
                    ChatCompletionRequestAssistantMessageContent::Array(parts) => {
                        info!("OpenAI final response: [Multipart Content]");
                        let mut combined_text = String::new();
                        for part in parts {
                            // *** Use correct import path for variants ***
                            if let ChatCompletionRequestAssistantMessageContentPart::Text(text_part) = part {
                                combined_text.push_str(&text_part.text);
                                combined_text.push('\n');
                            } else if let ChatCompletionRequestAssistantMessageContentPart::Refusal(refusal_part) = part {
                                combined_text.push_str(&format!("[Refusal Content]\n{refusal_part:?}"));
                            }
                            // Note: Need to handle ImageUrl part if expecting it in assistant responses
                        }
                        println!("\nAssistant:\n{}", combined_text.trim());
                    }
                }

                break; // Break inner loop, assistant gave final answer, wait for next user input

            } else {
                 warn!("OpenAI response had neither content nor tool calls.");
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
        max_tokens: Some(1024), // Add max_tokens for vision model
        ..Default::default()
    };

    // Call the API and log detailed errors
    let response = client.chat().create(request).await
        // *** Added detailed error logging using map_err ***
        .map_err(|e| {
            error!("Vision API call failed. Error details: {:?}", e);
             match &e {
                 OpenAIError::ApiError(api_error) => {
                      error!("--> Specific Vision API Error: {:?}", api_error);
                 }
                 OpenAIError::Reqwest(req_err) => {
                       error!("--> Specific Vision Network Error: {:?}", req_err);
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
