use anyhow::{Context, Result};
use async_openai::{
    types::{
        ChatCompletionRequestAssistantMessage, ChatCompletionRequestAssistantMessageContent,
        ChatCompletionRequestMessage,
        ChatCompletionRequestSystemMessage, ChatCompletionRequestSystemMessageContent,
        ChatCompletionRequestToolMessage, ChatCompletionRequestToolMessageContent,
        ChatCompletionRequestUserMessage, ChatCompletionRequestUserMessageContent,
        // Removed ChatCompletionMessageToolCall as it's unused
        ChatCompletionTool, ChatCompletionToolChoiceOption, ChatCompletionToolType,
        CreateChatCompletionRequest, FunctionObject,
        Role,
    },
    Client as OpenAIClient,
};
use rmcp::{
    // Removed ListToolsRequest, RawContent. Removed Content, Tool alias.
    model::{CallToolRequestParam, RawContent}, // Removed ListToolsRequest
    service::{RoleClient, RunningService}, // RoleClient likely needs feature flag
    serve_client, // serve_client likely needs feature flag
};
use serde_json::{json, Map, Value}; // Added Map
use std::{collections::VecDeque, env}; // Removed Cow import
use tokio::net::TcpSocket; // Use TcpSocket for connecting
use tracing::{debug, error, info, warn};

// Configuration
const MCP_SERVER_ADDR: &str = "127.0.0.1:9001"; // Address of your TCP MCP Server
const MAX_CONVERSATION_DEPTH: usize = 15; // Limit conversation history
const OPENAI_MODEL: &str = "gpt-4.1-mini"; // Or your preferred model

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
    // *** Call list_tools() without arguments ***
    let mcp_tools_result = mcp_peer
        .list_tools(None) // Call without arguments
        .await
        .context("Failed to list tools from MCP server")?;
    info!("Available tools: {:?}", mcp_tools_result.tools.iter().map(|t| &t.name).collect::<Vec<_>>());

    // Convert MCP tools to OpenAI tool format
    let openai_tools: Vec<ChatCompletionTool> = mcp_tools_result
        .tools
        .into_iter()
        .map(|mcp_tool| {
            let parameters_value = Some(Value::Object(mcp_tool.input_schema.as_ref().clone()));

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
    let system_prompt = "You are a helpful AI assistant. You can control the user's desktop using the available tools. Analyze the user's request and use the tools provided to fulfill it step-by-step. Ask for clarification if the request is ambiguous. Inform the user upon successful completion or if an error occurs.".to_string();
    conversation_history.push_back(ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage{
        content: ChatCompletionRequestSystemMessageContent::Text(system_prompt),
        name: None
    }));


    loop {
        // Get user input
        println!("\nEnter your request (or type 'quit'):");
        let mut user_input = String::new();
        std::io::stdin().read_line(&mut user_input)?;
        let user_input = user_input.trim();

        if user_input.eq_ignore_ascii_case("quit") {
            break;
        }
        if user_input.is_empty() {
            continue;
        }

        add_message_to_history(&mut conversation_history, ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessage {
                content: ChatCompletionRequestUserMessageContent::Text(user_input.to_string()),
                name: None,
            }
        ));

        // --- Call OpenAI Loop (Handles potential multi-step tool calls) ---
        loop {
            info!("Sending request to OpenAI...");
            debug!("Conversation History: {:?}", conversation_history);

            let request = CreateChatCompletionRequest {
                model: OPENAI_MODEL.to_string(),
                messages: conversation_history.iter().cloned().collect(),
                tools: if openai_tools.is_empty() { None } else { Some(openai_tools.clone()) },
                tool_choice: if openai_tools.is_empty() { None } else { Some(ChatCompletionToolChoiceOption::Auto) },
                ..Default::default()
            };

            let response = match openai_client.chat().create(request).await {
                 Ok(res) => res,
                 Err(e) => {
                     error!("OpenAI API error: {}", e);
                     println!("Error communicating with OpenAI. Please try again.");
                     conversation_history.pop_back(); // Allow retry
                     break; // Break inner loop
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

            // Construct the request assistant message from the response message
            let assistant_message_request = ChatCompletionRequestAssistantMessage {
                content: choice.message.content.clone().map(ChatCompletionRequestAssistantMessageContent::Text),
                name: None,
                tool_calls: choice.message.tool_calls.clone(),
                function_call: choice.message.function_call.clone(), // Deprecated but include
                audio: None,
                refusal: None,
            };
            add_message_to_history(&mut conversation_history, ChatCompletionRequestMessage::Assistant(assistant_message_request.clone()));


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
                            add_message_to_history(&mut conversation_history, ChatCompletionRequestMessage::Tool(ChatCompletionRequestToolMessage{
                                tool_call_id: call_id,
                                content: ChatCompletionRequestToolMessageContent::Text(error_msg)
                            }));
                            continue; // Skip this tool call
                        }
                    };

                    let arguments_map: Option<Map<String, Value>> = match arguments_value {
                        Value::Object(map) => Some(map),
                        _ => {
                            warn!("Tool arguments were not a JSON object for tool '{}'. Treating as empty.", tool_name);
                            None
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
                            match mcp_result.content.into_iter().next() { // Use .contents field
                                Some(content) => {
                                    match content.raw { // Access .raw field of Annotated<RawContent>
                                        RawContent::Text(raw_text_content) => {
                                            raw_text_content.text
                                        }
                                        RawContent::Image(_) => {
                                            json!({"result": format!("Tool '{}' returned image content.", mcp_request.name)}).to_string()
                                        }
                                        RawContent::Resource(_) => {
                                            json!({"result": format!("Tool '{}' returned resource content.", mcp_request.name)}).to_string()
                                        }
                                    }
                                }
                                None => {
                                    json!({"result": format!("Tool '{}' executed successfully but returned no content.", mcp_request.name)}).to_string()
                                }
                            }
                        }
                        Err(e) => {
                            error!("Failed to call MCP tool '{}': {}", mcp_request.name, e);
                            json!({
                                "error": format!("Failed to execute tool '{}' via MCP: {}", mcp_request.name, e)
                            }).to_string()
                        }
                    };


                    add_message_to_history(&mut conversation_history, ChatCompletionRequestMessage::Tool(ChatCompletionRequestToolMessage{
                        tool_call_id: call_id,
                        content: ChatCompletionRequestToolMessageContent::Text(tool_result_content_str)
                    }));
                }
                continue; // Continue the inner OpenAI loop

            } else if let Some(content) = assistant_message_request.content {
                // --- Handle Regular Assistant Message ---
                match content {
                    ChatCompletionRequestAssistantMessageContent::Text(text)=>{
                        info!("OpenAI final response: {}",text);
                        println!("\nAssistant:\n{}",text);
                    }
                    ChatCompletionRequestAssistantMessageContent::Array(parts) => { // Handle array content type
                        info!("OpenAI final response: [Multipart Content]");
                        // Process parts if needed, e.g., extract text
                        let mut combined_text = String::new();
                        for part in parts {
                             if let async_openai::types::ChatCompletionRequestAssistantMessageContentPart::Text(text_part) = part {
                                 combined_text.push_str(&text_part.text);
                                 combined_text.push('\n'); // Add separator if desired
                             } else if let async_openai::types::ChatCompletionRequestAssistantMessageContentPart::Refusal(refusal_part) = part {
                                 combined_text.push_str(&format!("[Refusal Content]\n{refusal_part:?}"));
                             }
                        }
                        println!("\nAssistant:\n{}", combined_text.trim());
                    }
                }

                break; // Break inner loop

            } else {
                 warn!("OpenAI response had neither content nor tool calls.");
                 println!("Assistant provided an empty response.");
                 break; // Break inner loop
            }
        } // End inner OpenAI loop

    } // End main user input loop

    info!("Exiting AI Client.");
    Ok(())
}


// Helper to manage conversation history size (remains the same)
fn add_message_to_history(history: &mut VecDeque<ChatCompletionRequestMessage>, message: ChatCompletionRequestMessage) {
     history.push_back(message);
     while history.len() > MAX_CONVERSATION_DEPTH {
         let role_at_index_1 = match history.get(1) {
             Some(ChatCompletionRequestMessage::System(_)) => Some(Role::System),
             Some(ChatCompletionRequestMessage::User(_)) => Some(Role::User),
             Some(ChatCompletionRequestMessage::Assistant(_)) => Some(Role::Assistant),
             Some(ChatCompletionRequestMessage::Tool(_)) => Some(Role::Tool),
             _ => None,
         };

         if history.len() > 1 && role_at_index_1 != Some(Role::System) {
             history.remove(1);
         } else {
             let role_at_front = match history.front() {
                 Some(ChatCompletionRequestMessage::System(_)) => Some(Role::System),
                 _ => None,
             };
             if role_at_front != Some(Role::System) {
                 history.pop_front();
             } else {
                 break;
             }
         }
     }
}
