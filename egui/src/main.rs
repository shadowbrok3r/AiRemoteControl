// Import anyhow macro
use anyhow::{anyhow, Context}; use base64::Engine;
// *** Add screenshots and base64 imports ***
use screenshots::Screen;
// *** Add display-info import ***
use display_info::DisplayInfo;
use enigo::{
    Button, Coordinate,
    Direction,
    Enigo, Key, Keyboard, Mouse, Settings,
};

// --- Specific rmcp Imports ---
use rmcp::schemars; // For deriving schema
use rmcp::handler::server::ServerHandler;
// use rmcp::transport::stdio;
use tokio::net::TcpListener; // Added TcpListener
// Import types needed for tool return values and ServerHandler impl
use rmcp::model::{
    // *** Added ErrorCode, ErrorData ***
    CallToolResult, Content, ErrorCode, ErrorData, Implementation, ProtocolVersion, ServerCapabilities,
    ServerInfo,
};
// Added serve_server back
// *** Ensure tool_box is imported ***
use rmcp::{serve_server, tool}; // Keep McpError for type alias if needed internally

use serde::{Deserialize, Serialize};
use serde_json::json;
use tracing::{info, warn}; // Added warn
use tracing_subscriber::EnvFilter; // Import EnvFilter for tracing setup

// --- Tool Parameter Struct Definitions ---

#[derive(Deserialize, Debug, Serialize, schemars::JsonSchema)]
struct MoveMouseParams {
    #[schemars(description = "Target X coordinate (absolute pixel value).")]
    x: i32,
    #[schemars(description = "Target Y coordinate (absolute pixel value).")]
    y: i32,
    #[schemars(description = "Type of mouse move ('Absolute'/'Abs' for absolute coordinates, 'Relative'/'Rel' for relative offset).")]
    coordinate: String
}

#[derive(Deserialize, Debug, Serialize, schemars::JsonSchema)]
struct MouseClickParams {
    #[schemars(description = "Which mouse button/action ('Left', 'Right', 'Middle', 'Back', 'Forward', 'ScrollUp', 'ScrollDown'). Case-insensitive.")]
    button: String, // Consider using an enum in production
    #[schemars(description = "Type of action ('Click', 'Press', 'Release'). Default is 'Click'. Double click not directly supported.", default)]
    click_type: Option<String>,
}


// *** Added consolidated KeyboardActionParams ***
#[derive(Deserialize, Debug, Serialize, schemars::JsonSchema)]
struct KeyboardActionParams {
    #[schemars(description = "Optional: Text to type using enigo's text input method.")]
    text: Option<String>,
    #[schemars(description = "Optional: A specific key to press/release/click (e.g., 'a', 'Enter', 'Control', 'Shift', 'Alt', 'F5', 'PageDown'). Takes precedence over 'text' if both are provided.")]
    key: Option<String>,
    #[schemars(description = "Action for the specified 'key': 'Click' (default), 'Press', 'Release'. Ignored if 'text' is used.", default)]
    key_action: Option<String>,
}

#[derive(Deserialize, Debug, Serialize, schemars::JsonSchema)]
struct CaptureScreenParams {
    #[schemars(description = "Optional X coordinate of the top-left corner for regional capture.")]
    x: Option<i32>,
    #[schemars(description = "Optional Y coordinate of the top-left corner for regional capture.")]
    y: Option<i32>,
    #[schemars(description = "Optional width for regional capture.")]
    width: Option<u32>,
    #[schemars(description = "Optional height for regional capture.")]
    height: Option<u32>,
    // Maybe add monitor index parameter later
}

// *** Added dummy struct for parameterless tool ***
#[derive(Deserialize, Debug, Serialize, schemars::JsonSchema)]
struct GetScreenDetailsParams;

// *** Added dummy struct for parameterless tool ***
#[derive(Deserialize, Debug, Serialize, schemars::JsonSchema)]
struct GetMousePositionParams;

#[derive(Deserialize, Debug, Serialize, schemars::JsonSchema)]
struct RunShellParams {
    command: String,
    args: Vec<String>,
}

// --- Tool Provider Implementation ---

#[derive(Clone)] // Clone is required by ServerHandler
struct DesktopToolProvider;

// *** First impl block: Contains the tool definitions ***
#[tool(tool_box)] // Apply tool_box here as well
impl DesktopToolProvider {

    // --- Screen Calibration ---
    #[tool(name = "get_screen_details", description = "Gets the primary screen resolution (width and height).")]
    // *** Restored user implementation, adjusted return type and error handling pattern ***
    async fn get_screen_details(
        &self,
        #[tool(param)] _params: GetScreenDetailsParams // Use dummy struct
    ) -> Result<CallToolResult, ErrorData> {
        info!("Received request to get screen details.");
        // Apply user's required error pattern
        let display_infos = DisplayInfo::all()
            .map_err(|e| anyhow!(e).context("display_info::DisplayInfo::all() failed")) // Map specific error -> anyhow
            .map_err(|e| ErrorData::new(ErrorCode::INTERNAL_ERROR, e.to_string(), None))?; // Map anyhow -> ErrorData

        let primary = display_infos
            .iter()
            .find(|d| d.is_primary)
            .or(display_infos.first());

        if let Some(primary_screen) = primary {
            let result_json = json!({
                "status": "success", // Indicate success
                "width": primary_screen.width,
                "height": primary_screen.height,
                "scale_factor": primary_screen.scale_factor,
                "x": primary_screen.x,
                "y": primary_screen.y
            });
            // Apply user's required error pattern to Content::json
            Ok(CallToolResult::success(vec![Content::json(result_json)
                .map_err(|e| anyhow!(e).context("Failed to serialize screen details to JSON"))
                .map_err(|e| ErrorData::new(ErrorCode::INTERNAL_ERROR, e.to_string(), None))?
            ]))
        } else {
            // Return ErrorData directly for logical errors
            Err(ErrorData::new(
                ErrorCode::INTERNAL_ERROR, // Or a more specific code like NotFound
                "Could not find primary or any display".to_string(),
                None
            ))
        }
    }

    // --- Mouse Tools ---
    #[tool(name = "move_mouse", description = "Moves the mouse cursor")]
     // *** Return Result<..., ErrorData> ***
    async fn move_mouse(
        &self,
        #[tool(aggr)] params: MoveMouseParams
    ) -> Result<CallToolResult, ErrorData> {
        info!("Executing move mouse to: {:?}", params);
        let mut enigo = Enigo::new(&Settings::default())
            .map_err(|e| ErrorData::new(ErrorCode::INTERNAL_ERROR, e.to_string(), None))?;

        let coordinate = match params.coordinate.to_lowercase().as_str() {
            "absolute" | "abs" => Coordinate::Abs,
            "relative" | "rel" | _ => Coordinate::Rel, // Default to Relative
        };

        if coordinate == Coordinate::Rel {
            info!("Moving mouse relatively by ({}, {})", params.x, params.y);
        } else {
             info!("Moving mouse absolutely to ({}, {})", params.x, params.y);
        }

        enigo.move_mouse(params.x, params.y, coordinate)
            .map_err(|e| ErrorData::new(ErrorCode::INTERNAL_ERROR, format!("Couldnt move mouse: {e:?}"), None))?;

        let (x, y) = enigo
            .location()
            .map_err(|e| ErrorData::new(ErrorCode::INTERNAL_ERROR, e.to_string(), None))?;

        info!("Mouse moved successfully.");
        // Apply user's required error pattern to Content::json
        Ok(CallToolResult::success(
            vec![
                Content::json(
                    json!({
                        "status": "success", // Indicate success
                        "current_x": x, // Return current position after move
                        "current_y": y
                    })
                )
                .map_err(|e| anyhow!(e).context("Failed to serialize move_mouse result"))
                .map_err(|e| ErrorData::new(ErrorCode::INTERNAL_ERROR, e.to_string(), None))?
            ]
        ))
    }

    // *** Added: get_mouse_position ***
    #[tool(name = "get_mouse_position", description = "Gets the current absolute screen coordinates (X, Y) of the mouse cursor")]
    // *** Return Result<..., ErrorData> ***
    async fn get_mouse_position(
        &self,
        #[tool(param)] _params: GetMousePositionParams,
    ) -> Result<CallToolResult, ErrorData> {
        info!("Executing get mouse position.");
        let enigo = Enigo::new(&Settings::default())
            .map_err(|e| ErrorData::new(ErrorCode::INTERNAL_ERROR, e.to_string(), None))?;

        let (x, y) = enigo
            .location()
            .map_err(|e| ErrorData::new(ErrorCode::INTERNAL_ERROR, e.to_string(), None))?;

        info!("Mouse position retrieved successfully: ({}, {})", x, y);
        let result_json = json!({
            "status": "success",
            "x": x,
            "y": y
        });
         // Apply user's required error pattern to Content::json
        Ok(CallToolResult::success(vec![Content::json(result_json)
            .map_err(|e| anyhow!(e).context("Failed to serialize get_mouse_position result"))
            .map_err(|e| ErrorData::new(ErrorCode::INTERNAL_ERROR, e.to_string(), None))?
        ]))
    }

    #[tool(name = "mouse_action", description = "Performs a mouse action (click, press, release) or scrolls the mouse wheel using enigo.")]
     // *** Return Result<..., ErrorData> ***
    async fn mouse_action(
        &self,
        #[tool(aggr)] params: MouseClickParams
    ) -> Result<CallToolResult, ErrorData> {
        info!("Executing mouse click: {:?}", params);
        let mut enigo = Enigo::new(&Settings::default())
            .map_err(|e| ErrorData::new(ErrorCode::INTERNAL_ERROR, e.to_string(), None))?;

        let button_str = params.button.to_lowercase();
        let action_str = params.click_type.as_deref().unwrap_or("click").to_lowercase();

        let direction = match action_str.as_str() {
            "click" => Direction::Click,
            "press" => Direction::Press,
            "release" => Direction::Release,
            "double" => {
                warn!("Double click not yet implemented, performing single click instead.");
                Direction::Click
            }
            _ => Direction::Click
        };

        let button_enum = match button_str.as_str() {
            "left" => Button::Left,
            "right" => Button::Right,
            "middle" => Button::Middle,
            "back" => Button::Back, // Keep Back/Forward if needed
            "forward" => Button::Forward,
            "scrollup" | "scroll_up" => Button::ScrollUp,
            "scrolldown" | "scroll_down" => Button::ScrollDown,
            "scrollleft" | "scroll_left" => Button::ScrollLeft, // Add scroll left/right
            "scrollright" | "scroll_right" => Button::ScrollRight,
            _ => return Err(ErrorData::invalid_params(
                format!("Invalid mouse button/action specified: '{}'. Use 'Left', 'Right', 'Middle', 'ScrollUp', 'ScrollDown', etc.", params.button),
                None
            )),
        };
        
        enigo
            .button(button_enum, direction)
            .map_err(|e| ErrorData::new(ErrorCode::INTERNAL_ERROR, e.to_string(), None))?;

        info!("Mouse clicked successfully: {}", button_str);

        // Apply user's required error pattern to Content::json
        Ok(CallToolResult::success(vec![Content::json(json!({ "status": "success", "button": button_str, "action": action_str }))
            .map_err(|e| anyhow!(e).context("Failed to serialize mouse_action result"))
            .map_err(|e| ErrorData::new(ErrorCode::INTERNAL_ERROR, e.to_string(), None))?
        ]))
    }

    #[tool(name = "keyboard_action", description = "Types text or performs a key event (click, press, release) using enigo.")]
    async fn keyboard_action(
        &self,
        #[tool(aggr)] params: KeyboardActionParams
    ) -> Result<CallToolResult, ErrorData> {
        info!("Executing keyboard action: {:?}", params);
        let mut enigo = Enigo::new(&Settings::default())
             .map_err(|e| ErrorData::new(ErrorCode::INTERNAL_ERROR, e.to_string(), None))?;

        // Prefer key action over typing text if both are provided
        if let Some(key_str) = &params.key {
            let action_str = params.key_action.as_deref().unwrap_or("click").to_lowercase();
            info!("Performing key action: key='{}', action='{}'", key_str, action_str);

            let direction = match action_str.as_str() {
                "click" => Direction::Click,
                "press" => Direction::Press,
                "release" => Direction::Release,
                 _ => {
                    warn!("Invalid key_action '{}', defaulting to Click.", action_str);
                    Direction::Click
                }
            };

            // Map string to enigo::Key
            // This requires a comprehensive mapping. Add common keys here.
            let key_enum = match key_str.to_lowercase().as_str() {
                "alt" => Key::Alt,
                "backspace" => Key::Backspace,
                "capslock" | "caps_lock" => Key::CapsLock,
                "control" | "ctrl" => Key::Control,
                "delete" => Key::Delete,
                "down" | "downarrow" => Key::DownArrow,
                "end" => Key::End,
                "escape" | "esc" => Key::Escape,
                "f1" => Key::F1, "f2" => Key::F2, "f3" => Key::F3, "f4" => Key::F4,
                "f5" => Key::F5, "f6" => Key::F6, "f7" => Key::F7, "f8" => Key::F8,
                "f9" => Key::F9, "f10" => Key::F10, "f11" => Key::F11, "f12" => Key::F12,
                "home" => Key::Home,
                "left" | "leftarrow" => Key::LeftArrow,
                "meta" | "win" | "command" | "super" | "windows" => Key::Meta,
                "option" => Key::Option, // Typically Alt on Windows/Linux, Option on macOS
                "pagedown" | "page_down" => Key::PageDown,
                "pageup" | "page_up" => Key::PageUp,
                "return" | "enter" => Key::Return,
                "right" | "rightarrow" => Key::RightArrow,
                "shift" => Key::Shift,
                "space" => Key::Space,
                "tab" => Key::Tab,
                "up" | "uparrow" => Key::UpArrow,
                // Handle single characters directly using Key::Layout
                s if s.chars().count() == 1 => Key::Unicode(s.chars().next().unwrap()),
                // Handle Unicode characters if needed (though enigo::text is better)
                // s if s.starts_with("U+") => Key::Unicode(...)
                _ => return Err(ErrorData::invalid_params(
                    format!("Unsupported key specified: '{}'.", key_str), None
                )),
            };

            enigo.key(key_enum, direction)
                 .map_err(|e| ErrorData::new(ErrorCode::INTERNAL_ERROR, e.to_string(), None))?;
            info!("Key action successful.");
            Ok(CallToolResult::success(vec![Content::json(json!({ "status": "success", "key": key_str, "action": action_str }))
                .map_err(|e| anyhow!(e).context("Failed to serialize keyboard key action result"))
                .map_err(|e| ErrorData::new(ErrorCode::INTERNAL_ERROR, e.to_string(), None))?
            ]))

        } else if let Some(text_to_type) = &params.text {
            info!("Typing text: '{}'", text_to_type);
            enigo.text(text_to_type)
                .map_err(|e| ErrorData::new(ErrorCode::INTERNAL_ERROR, e.to_string(), None))?;
            info!("Text typing successful.");
            Ok(CallToolResult::success(vec![Content::json(json!({ "status": "success", "text_typed": text_to_type }))
                .map_err(|e| anyhow!(e).context("Failed to serialize keyboard text typing result"))
                .map_err(|e| ErrorData::new(ErrorCode::INTERNAL_ERROR, e.to_string(), None))?
            ]))
        } else {
            Err(ErrorData::invalid_params(
                "Keyboard action requires either 'key' or 'text' parameter.".to_string(), None
            ))
        }
    }

    // --- Screen Capture ---
    #[tool(name = "capture_screen", description = "Captures the screen (or a region) and returns image data as base64.")]
    async fn capture_screen(
        &self,
        #[tool(aggr)] params: CaptureScreenParams
    ) -> Result<CallToolResult, ErrorData> {
        info!("Executing screen capture with params: {:?}", params);

        // *** Apply user's required error pattern ***
        let screens = Screen::all()
            .context("Failed to get screen list")
            .map_err(|e| ErrorData::new(ErrorCode::INTERNAL_ERROR, e.to_string(), None))?;

        let screen_to_capture = screens.first()
            .ok_or_else(|| anyhow!("No screen found to capture")) // Creates anyhow::Error
            .map_err(|e| ErrorData::new(ErrorCode::INTERNAL_ERROR, e.to_string(), None))?; // Map anyhow -> ErrorData
        info!("Capturing from screen ID: {}", screen_to_capture.display_info.id);


        let image = if let (Some(x), Some(y), Some(w), Some(h)) = (params.x, params.y, params.width, params.height) {
             info!("Capturing region: x={}, y={}, width={}, height={}", x, y, w, h);
             // *** Apply user's required error pattern ***
             screen_to_capture.capture_area(x, y, w, h)
                .context("Failed to capture screen area") // Creates anyhow::Error
                .map_err(|e| ErrorData::new(ErrorCode::INTERNAL_ERROR, e.to_string(), None))? // Map anyhow -> ErrorData
        } else {
             info!("Capturing full screen.");
             // *** Apply user's required error pattern ***
             screen_to_capture.capture()
                .context("Failed to capture full screen") // Creates anyhow::Error
                .map_err(|e| ErrorData::new(ErrorCode::INTERNAL_ERROR, e.to_string(), None))? // Map anyhow -> ErrorData
        };
        info!("Capture successful ({}x{})", image.width(), image.height());

        let base64_image = base64::engine::general_purpose::STANDARD.encode(image.as_raw());
        info!("Encoded image to base64 (length: {})", base64_image.len());


        let result_json = json!({
            "status": "success",
            "format": "png",
            "width": image.width(),
            "height": image.height(),
            "base64_data": base64_image,
        });
        // Apply user's required error pattern to Content::json
        Ok(CallToolResult::success(vec![Content::json(result_json)
            .map_err(|e| anyhow!(e).context("Failed to serialize capture_screen result"))
            .map_err(|e| ErrorData::new(ErrorCode::INTERNAL_ERROR, e.to_string(), None))?
        ]))
    }

    // --- Shell Command ---
    #[tool(name = "run_shell_command", description = "Runs a command in the default system shell.")]
     async fn run_shell_command(
        &self,
        #[tool(aggr)] params: RunShellParams
    ) -> Result<CallToolResult, ErrorData> {
        info!("Received request to run command: {:?}", params);

        // *** Apply user's required error pattern ***
        let output = tokio::process::Command::new(&params.command)
            .args(&params.args)
            .kill_on_drop(true)
            .output()
            .await
            .context(format!("Failed to execute command: {}", params.command)) // Creates anyhow::Error
            .map_err(|e| ErrorData::new(ErrorCode::INTERNAL_ERROR, e.to_string(), None))?; // Map anyhow -> ErrorData

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let exit_code = output.status.code().unwrap_or(-1);

        info!(
            "Command '{}' executed. Status: {}, Stdout len: {}, Stderr len: {}",
            params.command, exit_code, stdout.len(), stderr.len()
        );

        let result_json = json!({
            "status": if output.status.success() { "success" } else { "error" },
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr,
        });

        // Apply user's required error pattern to Content::json
        Ok(CallToolResult::success(vec![Content::json(result_json)
             .map_err(|e| anyhow!(e).context("Failed to serialize run_shell_command result"))
             .map_err(|e| ErrorData::new(ErrorCode::INTERNAL_ERROR, e.to_string(), None))?
        ]))
    }
}

#[tool(tool_box)]
impl ServerHandler for DesktopToolProvider {
    // Provide basic server information
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::V_2024_11_05,
            capabilities: ServerCapabilities::builder()
                .enable_tools()
                .build(),
            server_info: Implementation::from_build_env(),
            instructions: Some("This server allows controlling the desktop via various tools (mouse, keyboard, screen capture, shell commands).".to_string()),
        }
    }
    // Add other ServerHandler methods if needed
}


// --- Main Function (Using TCP) ---

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive(tracing::Level::INFO.into()))
        .with_writer(std::io::stderr)
        .with_ansi(true)
        .init();

    // Spawn the TCP server task
    tokio::spawn(async move {
        if let Err(e) = run_mcp_server_tcp().await {
            tracing::error!("MCP Server error: {:?}", e);
        }
    });

    info!("Main thread running. MCP Server spawned in background. Press Ctrl+C to exit.");
    tokio::signal::ctrl_c().await?;
    info!("Ctrl+C received, shutting down.");

    Ok(())
}

// --- TCP Server Function ---
async fn run_mcp_server_tcp() -> anyhow::Result<()> {
    let addr = "127.0.0.1:9001"; // The TCP address to listen on
    let listener = TcpListener::bind(addr).await?;
    info!("MCP Server listening on TCP {}", addr);

    let tool_provider = DesktopToolProvider; // Create the tool provider instance

    loop {
        let (stream, client_addr) = listener.accept().await?;
        info!("Accepted TCP connection from: {}", client_addr);
        let provider_clone = tool_provider.clone();

        tokio::spawn(async move {
            info!("Serving client {}...", client_addr);
            match serve_server(provider_clone, stream).await {
                Ok(server_handle) => {
                    if let Err(e) = server_handle.waiting().await {
                        if !e.to_string().contains("connection closed")
                            && !e.to_string().contains("Connection reset by peer")
                            && !e.to_string().contains("broken pipe")
                           {
                            tracing::error!("Client {} error: {:?}", client_addr, e);
                        } else {
                            info!("Client {} disconnected.", client_addr);
                        }
                    }
                }
                Err(e) => {
                    tracing::error!("Failed to start serving client {}: {:?}", client_addr, e);
                }
            }
        });
    }
    // Ok(()) // Unreachable
}
