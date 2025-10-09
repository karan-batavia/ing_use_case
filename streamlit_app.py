import streamlit as st
import requests
import io
import uuid
import os
import json
from typing import Optional, Dict, Any


# API Configuration
# Automatically detect if running in Docker or local development
def get_api_base_url():
    """
    Determine the correct API base URL based on environment.
    - Docker: Use service name from docker-compose
    - Local: Use localhost
    """
    # Check if running in Docker by looking for environment variables or Docker-specific files
    if os.path.exists("/.dockerenv") or os.environ.get("DOCKER_ENV") == "true":
        # Running in Docker - use service name from docker-compose.yml
        return "http://fastapi-app:8000"  # 'fastapi-app' matches the service name in docker-compose.yml
    else:
        # Local development
        return "http://localhost:8000"


API_BASE_URL = get_api_base_url()
API_ENDPOINTS = {
    "login": f"{API_BASE_URL}/auth/login",
    "redact": f"{API_BASE_URL}/redact",
    "predict": f"{API_BASE_URL}/predict",
    "descrub": f"{API_BASE_URL}/de-scrub",
    "scrub_file_download": f"{API_BASE_URL}/scrub-file-download",
    "health": f"{API_BASE_URL}/health",
    "session_status": f"{API_BASE_URL}/auth/session-status",
}

# Session state keys for authentication
AUTH_TOKEN_KEY = "api_token"
USER_INFO_KEY = "user_info"


def get_auth_headers() -> Dict[str, str]:
    """Get authentication headers for API requests."""
    token = st.session_state.get(AUTH_TOKEN_KEY)
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def get_user_info_from_api() -> tuple[bool, Dict[str, Any]]:
    """
    Get current user information from the API.
    Since there's no dedicated user profile endpoint, we'll use the /redact endpoint
    with dummy data to trigger the get_current_user dependency and extract user info
    from the error response if needed.

    Returns:
        tuple: (success: bool, user_info: dict)
    """
    try:
        headers = get_auth_headers()

        # First try session-status (it should work but doesn't return admin status)
        response = requests.get(
            API_ENDPOINTS["session_status"], headers=headers, timeout=10
        )

        if response.status_code == 200:
            session_data = response.json()

            # The session-status doesn't include admin status, so we need to check
            # the JWT token payload for user_id and make educated guess or
            # try to decode the information from the token
            jwt_payload = get_current_user_details()

            # Try to make a simple API call that would reveal admin status
            # by checking if the user can access admin-only features
            try:
                # Try to access an admin endpoint to test admin status
                admin_test_response = requests.get(
                    f"{API_BASE_URL}/stats/sessions?limit=1", headers=headers, timeout=5
                )
                is_admin = admin_test_response.status_code != 403
            except:
                is_admin = False

            return True, {
                "user_id": session_data.get("user_id"),
                "session_id": session_data.get("session_id"),
                "is_admin": is_admin,
                "email": jwt_payload.get("sub", ""),  # 'sub' contains the email
            }
        else:
            return False, {"error": f"API error: {response.status_code}"}

    except Exception as e:
        return False, {"error": f"Request failed: {str(e)}"}


def get_current_user_details() -> Dict[str, Any]:
    """
    Get detailed user information including admin status from JWT token payload.
    This decodes the JWT token to extract user details without making an API call.
    """
    try:
        import base64
        import json

        token = st.session_state.get(AUTH_TOKEN_KEY)
        if not token:
            return {}

        # Decode JWT payload (middle part of the token)
        # JWT format: header.payload.signature
        parts = token.split(".")
        if len(parts) != 3:
            return {}

        # Decode the payload (add padding if needed for base64)
        payload = parts[1]
        # Add padding if needed
        padding = len(payload) % 4
        if padding:
            payload += "=" * (4 - padding)

        decoded_bytes = base64.urlsafe_b64decode(payload)
        payload_data = json.loads(decoded_bytes.decode("utf-8"))

        return payload_data

    except Exception as e:
        print(f"Error decoding JWT: {e}")
        return {}


def login_to_api(email: str, password: str) -> tuple[bool, str, Dict[str, Any]]:
    """
    Login to the API and return success status, message, and user info.

    Returns:
        tuple: (success: bool, message: str, user_info: dict)
    """
    try:
        # Prepare login data as JSON for the API
        # The API expects 'email' field, so we use username as email
        login_data = {"email": email, "password": password}  # API expects 'email' field

        # Set proper headers for JSON request
        headers = {"Content-Type": "application/json"}

        response = requests.post(
            API_ENDPOINTS["login"],
            json=login_data,  # Send as JSON instead of form data
            headers=headers,
            timeout=10,
        )

        if response.status_code == 200:
            token_data = response.json()

            # Store the token temporarily to make the session status call
            temp_token = token_data.get("access_token")
            st.session_state[AUTH_TOKEN_KEY] = temp_token

            # Get real user information including admin status
            success_user_info, user_details = get_user_info_from_api()
            if not success_user_info:
                # Fallback: try to extract from JWT token
                jwt_payload = get_current_user_details()
                user_details = {"user_id": jwt_payload.get("user_id", email)}

            return (
                True,
                "Login successful!",
                {
                    "token": temp_token,
                    "user_id": token_data.get("user_id", email),
                    "username": email,  # Use email as username for display
                    "role": "admin" if user_details.get("is_admin", False) else "user",
                    "is_admin": user_details.get("is_admin", False),
                    "session_id": token_data.get("session_id"),
                },
            )
        else:
            error_detail = "Unknown error"
            try:
                error_data = response.json()
                if isinstance(error_data, dict):
                    error_detail = error_data.get("detail", str(error_data))
                else:
                    error_detail = str(error_data)
            except:
                error_detail = response.text
            return False, f"Login failed: {error_detail}", {}

    except requests.exceptions.RequestException as e:
        return False, f"Connection error: {str(e)}", {}
    except Exception as e:
        return False, f"Login error: {str(e)}", {}


def call_predict_api(redacted_text: str) -> tuple[bool, Dict[str, Any]]:
    """
    Call the /predict API endpoint to send redacted text to Gemini and get a response.
    This simulates what would happen if someone tried to use the redacted text.

    Returns:
        tuple: (success: bool, response_data: dict)
    """
    try:
        headers = get_auth_headers()
        headers["Content-Type"] = "application/json"

        payload = {"prompt": redacted_text}

        # Debug information
        st.write("🔍 **Debug Info:**")
        st.write(f"- API URL: `{API_ENDPOINTS['predict']}`")
        st.write(f"- Payload: `{payload}`")
        st.write(f"- Headers: `{headers}`")

        response = requests.post(
            API_ENDPOINTS["predict"], json=payload, headers=headers, timeout=30
        )

        st.write(f"- Response Status: `{response.status_code}`")
        st.write(f"- Response Text: `{response.text[:500]}...`")

        if response.status_code == 200:
            return True, response.json()
        else:
            return False, {
                "error": f"API error: {response.status_code}",
                "detail": response.text,
                "full_response": response.text,
            }

    except Exception as e:
        return False, {"error": f"Request failed: {str(e)}"}


def call_descrub_api(
    session_id: str, scrubbed_text: str
) -> tuple[bool, Dict[str, Any]]:
    """
    Call the /de-scrub API endpoint to restore original text using session_id and scrubbed_text.

    Returns:
        tuple: (success: bool, response_data: dict)
    """
    try:
        headers = get_auth_headers()
        headers["Content-Type"] = "application/json"

        payload = {"session_id": session_id, "scrubbed_text": scrubbed_text}

        response = requests.post(
            API_ENDPOINTS["descrub"], json=payload, headers=headers, timeout=30
        )

        if response.status_code == 200:
            return True, response.json()
        else:
            return False, {
                "error": f"API error: {response.status_code}",
                "detail": response.text,
            }

    except Exception as e:
        return False, {"error": f"Request failed: {str(e)}"}


def call_logs_api(limit: int = 50) -> tuple[bool, Dict[str, Any]]:
    """
    Call the API to fetch audit logs for admin users.
    
    Returns:
        tuple: (success: bool, logs_data: dict)
    """
    try:
        headers = get_auth_headers()
        
        # Try to get logs from a stats endpoint or similar
        response = requests.get(
            f"{API_BASE_URL}/stats/sessions?limit={limit}", 
            headers=headers, 
            timeout=30
        )
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, {
                "error": f"API error: {response.status_code}",
                "detail": response.text,
            }
            
    except Exception as e:
        return False, {"error": f"Request failed: {str(e)}"}


def call_redact_api(
    text: str, sensitivity_level: Optional[str] = None
) -> tuple[bool, Dict[str, Any]]:
    """
    Call the /redact API endpoint.

    Returns:
        tuple: (success: bool, response_data: dict)
    """
    try:
        headers = get_auth_headers()
        headers["Content-Type"] = "application/json"

        payload = {"text": text}
        if sensitivity_level:
            payload["sensitivity_level"] = sensitivity_level

        response = requests.post(
            API_ENDPOINTS["redact"], json=payload, headers=headers, timeout=30
        )

        if response.status_code == 200:
            return True, response.json()
        else:
            return False, {
                "error": f"API error: {response.status_code}",
                "detail": response.text,
            }

    except Exception as e:
        return False, {"error": f"Request failed: {str(e)}"}


def call_scrub_file_download_api(
    file_bytes: bytes, filename: str, sensitivity_level: Optional[str] = None
) -> tuple[bool, bytes, str, str]:
    """
    Call the /scrub-file-download API endpoint.

    Returns:
        tuple: (success: bool, file_bytes: bytes, filename: str, content_type: str)
    """
    try:
        headers = get_auth_headers()

        files = {"file": (filename, file_bytes)}
        data = {}
        if sensitivity_level:
            data["sensitivity_level"] = sensitivity_level

        response = requests.post(
            API_ENDPOINTS["scrub_file_download"],
            files=files,
            data=data,
            headers=headers,
            timeout=60,
        )

        if response.status_code == 200:
            # Extract filename from Content-Disposition header
            content_disposition = response.headers.get("Content-Disposition", "")
            scrubbed_filename = filename
            if "filename=" in content_disposition:
                scrubbed_filename = content_disposition.split("filename=")[1].strip('"')

            content_type = response.headers.get(
                "Content-Type", "application/octet-stream"
            )

            return True, response.content, scrubbed_filename, content_type
        else:
            return (
                False,
                b"",
                "",
                f"API error: {response.status_code} - {response.text}",
            )

    except Exception as e:
        return False, b"", "", f"Request failed: {str(e)}"


def login_page():
    """Display login page with API authentication."""
    st.set_page_config(
        page_title="ING Prompt Scrubber - Login", page_icon="🔐", layout="centered"
    )

    # ING Brand Styling for Login Page
    st.markdown(
        """
    <style>
    /* Import ING-style fonts */
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@300;400;600;700&display=swap');
    
    /* Root variables for ING brand colors */
    :root {
        --ing-orange: #FF6200;
        --ing-orange-light: #FF7A1F;
        --ing-blue: #003F6C;
        --ing-white: #FFFFFF;
        --ing-light-gray: #F5F5F5;
        --ing-text-dark: #333333;
    }
    
    .main > div {
        background: linear-gradient(135deg, var(--ing-light-gray), var(--ing-white));
        min-height: 100vh;
        font-family: 'Source Sans Pro', sans-serif;
    }
    
    h1 {
        color: var(--ing-blue) !important;
        font-family: 'Source Sans Pro', sans-serif !important;
        font-weight: 700 !important;
        text-align: center !important;
    }
    
    h3 {
        color: var(--ing-orange) !important;
        font-family: 'Source Sans Pro', sans-serif !important;
        text-align: center !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, var(--ing-orange), var(--ing-orange-light)) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-family: 'Source Sans Pro', sans-serif !important;
        font-weight: 600 !important;
        width: 100% !important;
        padding: 0.75rem 1.5rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 4px rgba(255, 98, 0, 0.2) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 8px rgba(255, 98, 0, 0.3) !important;
    }
    
    .login-container {
        background: var(--ing-white);
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 8px 16px rgba(0, 63, 108, 0.1);
        border-top: 4px solid var(--ing-orange);
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        st.title("🔐 ING Prompt Scrubber")
        st.markdown("### Secure Document Processing Pipeline")
        st.markdown(
            '<div style="background: linear-gradient(90deg, var(--ing-orange), var(--ing-orange-light)); height: 2px; border-radius: 1px; margin: 1rem 0;"></div>',
            unsafe_allow_html=True,
        )

        with st.container():
            st.subheader("Authentication")
            st.markdown("Please enter your credentials to access the secure API:")

            # Email and password inputs
            email = st.text_input("Email:", placeholder="Enter your email address")
            password = st.text_input(
                "Password:", type="password", placeholder="Enter your password"
            )

            # Demo credentials info
            with st.expander("📋 Demo Credentials"):
                st.info(
                    """
                **For demonstration purposes:**
                - Admin user: `admin@ing.com` / `admin123`
                - Regular user: `user@ing.com` / `user123`
                
                **Features:**
                - ✅ Secure API authentication
                - ✅ Professional document redaction
                - ✅ Format preservation (PDF, DOCX, TXT, HTML, Images)
                - ✅ Visual redaction for PDFs and images
                - ✅ Banking compliance audit logging
                """
                )

            # Login button
            if st.button("🚀 Login", type="primary", use_container_width=True):
                if email and password:
                    with st.spinner("Authenticating..."):
                        success, message, user_info = login_to_api(email, password)

                    if success:
                        # Store authentication info in session
                        st.session_state[AUTH_TOKEN_KEY] = user_info["token"]
                        st.session_state[USER_INFO_KEY] = user_info
                        st.session_state.logged_in = True
                        st.session_state.user_role = user_info["role"]
                        st.session_state.username = user_info["username"]

                        st.success(f"✅ {message}")
                        st.success(
                            f"Welcome, {user_info['username']} ({user_info['role']})!"
                        )
                        st.balloons()
                        st.rerun()
                    else:
                        st.error(f"❌ {message}")

                        # Show environment-specific fallback option
                        if (
                            os.path.exists("/.dockerenv")
                            or os.environ.get("DOCKER_ENV") == "true"
                        ):
                            st.warning(
                                "💡 **Tip:** Make sure all Docker services are running: `docker-compose up`"
                            )
                        else:
                            st.warning(
                                f"💡 **Tip:** Make sure the API server is running: `uvicorn api:app --reload`"
                            )
                            st.caption(f"API URL: {API_BASE_URL}")

                        if st.button("🔧 Enable Demo Mode (No API)", key="demo_mode"):
                            # Set demo mode for testing UI without API
                            st.session_state.logged_in = True
                            st.session_state.user_role = (
                                "admin" if "admin" in email.lower() else "user"
                            )
                            st.session_state.username = email or "demo_user"
                            st.session_state.demo_mode = True
                            st.rerun()
                else:
                    st.error("Please enter both email and password.")

        # API Status indicator
        st.markdown("---")
        with st.expander("🔌 API Connection Status"):
            st.caption(f"**API URL:** `{API_BASE_URL}`")

            try:
                response = requests.get(f"{API_BASE_URL}/health", timeout=5)
                if response.status_code == 200:
                    st.success("✅ API server is online")
                else:
                    st.warning(
                        f"⚠️ API server responded with status {response.status_code}"
                    )
            except:
                st.error("❌ API server is not accessible")

                # Show environment-specific help
                if (
                    os.path.exists("/.dockerenv")
                    or os.environ.get("DOCKER_ENV") == "true"
                ):
                    st.caption(
                        "Running in Docker - make sure both services are up: `docker-compose up`"
                    )
                else:
                    st.caption(
                        "Running locally - make sure the API server is running: `uvicorn api:app --reload`"
                    )

        st.markdown("</div>", unsafe_allow_html=True)  # Close login-container


def logout():
    """Handle user logout and clear API session."""
    # Clear all authentication and session data
    for key in [
        "logged_in",
        "user_role",
        "username",
        AUTH_TOKEN_KEY,
        USER_INFO_KEY,
        "demo_mode",
    ]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()


def main_app():
    """Main application using API endpoints."""
    st.set_page_config(page_title="ING Prompt Scrubber", page_icon="🔍", layout="wide")

    # ING Brand Styling
    st.markdown(
        """
    <style>
    /* Import ING-style fonts */
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@300;400;600;700&display=swap');
    
    /* Root variables for ING brand colors */
    :root {
        --ing-orange: #FF6200;
        --ing-orange-light: #FF7A1F;
        --ing-orange-dark: #E55100;
        --ing-blue: #003F6C;
        --ing-blue-light: #0066A8;
        --ing-blue-dark: #002C4C;
        --ing-gray: #767676;
        --ing-light-gray: #F5F5F5;
        --ing-white: #FFFFFF;
        --ing-text-dark: #333333;
    }
    
    /* Main app styling */
    .main > div {
        padding-top: 2rem;
        font-family: 'Source Sans Pro', sans-serif;
    }
    
    /* Header styling */
    h1 {
        color: var(--ing-blue) !important;
        font-family: 'Source Sans Pro', sans-serif !important;
        font-weight: 700 !important;
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    h2 {
        color: var(--ing-blue) !important;
        font-family: 'Source Sans Pro', sans-serif !important;
        font-weight: 600 !important;
        margin-top: 2rem !important;
    }
    
    h3 {
        color: var(--ing-orange) !important;
        font-family: 'Source Sans Pro', sans-serif !important;
        font-weight: 600 !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--ing-orange), var(--ing-orange-light)) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-family: 'Source Sans Pro', sans-serif !important;
        font-weight: 600 !important;
        padding: 0.75rem 1.5rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 4px rgba(255, 98, 0, 0.2) !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, var(--ing-orange-dark), var(--ing-orange)) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 8px rgba(255, 98, 0, 0.3) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0) !important;
    }
    
    /* Secondary button styling */
    .stButton > button[kind="secondary"] {
        background: linear-gradient(135deg, var(--ing-blue), var(--ing-blue-light)) !important;
        box-shadow: 0 2px 4px rgba(0, 63, 108, 0.2) !important;
    }
    
    .stButton > button[kind="secondary"]:hover {
        background: linear-gradient(135deg, var(--ing-blue-dark), var(--ing-blue)) !important;
        box-shadow: 0 4px 8px rgba(0, 63, 108, 0.3) !important;
    }
    
    /* Input fields styling */
    .stTextInput > div > div > input {
        border: 2px solid var(--ing-light-gray) !important;
        border-radius: 8px !important;
        font-family: 'Source Sans Pro', sans-serif !important;
        transition: border-color 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--ing-orange) !important;
        box-shadow: 0 0 0 2px rgba(255, 98, 0, 0.1) !important;
    }
    
    .stTextArea > div > div > textarea {
        border: 2px solid var(--ing-light-gray) !important;
        border-radius: 8px !important;
        font-family: 'Source Sans Pro', sans-serif !important;
        transition: border-color 0.3s ease !important;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: var(--ing-orange) !important;
        box-shadow: 0 0 0 2px rgba(255, 98, 0, 0.1) !important;
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        border: 2px dashed var(--ing-orange) !important;
        border-radius: 12px !important;
        background: linear-gradient(45deg, rgba(255, 98, 0, 0.03), rgba(255, 98, 0, 0.08)) !important;
        padding: 2rem !important;
        text-align: center !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, var(--ing-blue), var(--ing-blue-dark)) !important;
    }
    
    .css-1d391kg .css-1lcbmhc {
        color: var(--ing-white) !important;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--ing-light-gray) !important;
        border-radius: 8px !important;
        padding: 0.25rem !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 6px !important;
        color: var(--ing-gray) !important;
        font-family: 'Source Sans Pro', sans-serif !important;
        font-weight: 600 !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--ing-orange) !important;
        color: var(--ing-white) !important;
    }
    
    /* Metrics and info boxes */
    .stMetric {
        background: linear-gradient(135deg, var(--ing-white), var(--ing-light-gray)) !important;
        border-left: 4px solid var(--ing-orange) !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Success/Error messages */
    .stAlert {
        border-radius: 8px !important;
        font-family: 'Source Sans Pro', sans-serif !important;
    }
    
    .stAlert[data-baseweb="notification"] {
        border-left: 4px solid var(--ing-orange) !important;
    }
    
    /* Code blocks */
    .stCode {
        background: var(--ing-light-gray) !important;
        border: 1px solid var(--ing-gray) !important;
        border-radius: 8px !important;
        font-family: 'Fira Code', monospace !important;
    }
    
    /* Progress bars */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--ing-orange), var(--ing-orange-light)) !important;
    }
    
    /* Footer styling */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background: var(--ing-blue);
        color: var(--ing-white);
        text-align: center;
        padding: 0.5rem;
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 0.8rem;
        z-index: 999;
    }
    
    /* Custom ING container */
    .ing-container {
        background: linear-gradient(135deg, var(--ing-white), var(--ing-light-gray));
        border: 1px solid var(--ing-orange);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(255, 98, 0, 0.1);
    }
    
    /* ING brand accent */
    .ing-accent {
        background: linear-gradient(135deg, var(--ing-orange), var(--ing-orange-light));
        height: 4px;
        border-radius: 2px;
        margin: 1rem 0;
    }
    
    /* Animation for loading states */
    @keyframes ing-pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .ing-loading {
        animation: ing-pulse 1.5s ease-in-out infinite;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # ING Brand Header Accent
    st.markdown('<div class="ing-accent"></div>', unsafe_allow_html=True)

    # Header with user info and logout
    col1, col2 = st.columns([3, 1])

    with col1:
        user_info = st.session_state.get(USER_INFO_KEY, {})
        username = user_info.get("username", st.session_state.get("username", "User"))
        role = user_info.get("role", st.session_state.get("user_role", "user"))

        if st.session_state.get("demo_mode"):
            st.title("🔍 ING Prompt Scrubber (Demo Mode)")
            st.caption("⚠️ Running in demo mode - API features limited")
        else:
            st.title("🔍 ING Prompt Scrubber")
            st.caption(f"👤 Welcome, {username} ({role.title()})")

    with col2:
        if st.button("🚪 Logout", key="logout_btn"):
            logout()

    # Create two columns for layout
    col1, col2 = st.columns([1, 1])

    # Check if user is admin for additional tabs
    user_info = st.session_state.get(USER_INFO_KEY, {})
    is_admin = user_info.get("is_admin", False)

    with col1:
        st.markdown('<div class="ing-container">', unsafe_allow_html=True)
        st.header("📝 Input Options")

        # Create tabs - add Logs tab for admins
        if is_admin:
            tab1, tab2, tab3 = st.tabs(["Text Input", "File Upload", "📊 Admin Logs"])
        else:
            tab1, tab2 = st.tabs(["Text Input", "File Upload"])
            tab3 = None  # Initialize tab3 for non-admins

        with tab1:
            st.subheader("Text Redaction Workflow")
            text_input = st.text_area(
                "Enter your text here:",
                placeholder="Type or paste your text here...",
                height=200,
            )

            # Sensitivity level selector for text
            sensitivity_level = st.selectbox(
                "Classification Level:",
                options=[ "C1", "C2", "C3", "C4"],
                format_func=lambda x: f"Level {x}",
                help="C1: Public, C2: Internal, C3: Confidential, C4: Restricted",
            )

            # Step 1: Redact button
            if st.button(
                "🔍 Step 1: Redact Text", key="redact_text_btn", type="primary"
            ):
                if text_input:
                    with st.spinner("Processing text through /redact endpoint..."):
                        if st.session_state.get("demo_mode"):
                            # Demo mode fallback
                            processed_content = {
                                "success": True,
                                "original_text": text_input,
                                "redacted_text": "[DEMO] Text would be processed via API",
                                "detections": [],
                                "total_redacted": 0,
                                "session_id": "demo_session",
                            }
                        else:
                            # Use real API
                            success, api_response = call_redact_api(
                                text_input, sensitivity_level
                            )
                            if success:
                                processed_content = api_response
                                processed_content["original_text"] = text_input

                                # Store redaction result in session state for next steps
                                st.session_state.redaction_result = processed_content
                                st.success(
                                    "✅ Text redacted successfully! See results in the right panel."
                                )
                            else:
                                st.error(
                                    f"❌ API Error: {api_response.get('error', 'Unknown error')}"
                                )
                else:
                    st.warning("Please enter some text to redact.")

            # Show current redaction status
            if hasattr(st.session_state, "redaction_result"):
                redaction_data = st.session_state.redaction_result
                st.info(
                    f"✅ Text redacted | Session ID: `{redaction_data.get('session_id', 'N/A')}`"
                )

                # Step 2: Test redacted content with Gemini
                if st.button(
                    "� Step 2: Send to Gemini AI",
                    key="predict_btn",
                    type="secondary",
                ):
                    original_text = redaction_data.get("original_text", "")
                    redacted_text = redaction_data.get("redacted_text", "")
                    if redacted_text:
                        with st.spinner(
                            "Sending redacted text to Gemini AI..."
                        ):
                            if st.session_state.get("demo_mode"):
                                predict_result = {
                                    "success": True,
                                    "prediction": "[DEMO] This is a simulated Gemini response to the redacted text. In production, Gemini would receive the redacted content and generate a real response.",
                                    "model_used": "gemini-pro (demo)",
                                }
                            else:
                                success, predict_result = call_predict_api(
                                    redacted_text
                                )
                                if not success:
                                    st.error(
                                        f"❌ Prediction Error: {predict_result.get('error', 'Unknown error')}"
                                    )
                                    predict_result = None

                            if predict_result:
                                st.session_state.prediction_result = predict_result
                                st.success(
                                    "✅ Gemini response received! See results in the right panel."
                                )

                # Step 3: Admin-only descrub button
                user_info = st.session_state.get(USER_INFO_KEY, {})
                is_admin = user_info.get("is_admin", False)

                if is_admin:
                    st.markdown("---")
                    st.markdown("**🔧 Admin Functions**")
                    
                    # Descrub options
                    st.subheader("🔓 Step 3: De-scrub Options (Admin Only)")
                    
                    # Check what's available for descrubbing
                    has_redacted_text = bool(redaction_data.get("redacted_text", ""))
                    has_gemini_response = hasattr(st.session_state, "prediction_result") and st.session_state.prediction_result.get("success", False)
                    
                    if has_redacted_text or has_gemini_response:
                        descrub_options = []
                        if has_redacted_text:
                            descrub_options.append("Original Input Text")
                        if has_gemini_response:
                            descrub_options.append("Gemini Response")
                        
                        if len(descrub_options) > 1:
                            descrub_choice = st.radio(
                                "What would you like to de-scrub?",
                                options=descrub_options,
                                help="• Original Input Text: Restore the original input before redaction\n• Gemini Response: Generate what Gemini would respond with the original (unredacted) text"
                            )
                        else:
                            descrub_choice = descrub_options[0]
                            st.write(f"**Available for de-scrubbing:** {descrub_choice}")
                        
                        # Determine what text to descrub based on choice
                        if descrub_choice == "Original Input Text":
                            text_to_descrub = redaction_data.get("redacted_text", "")
                            descrub_type = "original"
                        else:  # Gemini Response
                            gemini_response = st.session_state.prediction_result.get("prediction", "")
                            text_to_descrub = gemini_response
                            descrub_type = "gemini"
                        
                        if st.button(
                            f"🔓 De-scrub {descrub_choice}",
                            key="descrub_btn",
                            type="secondary",
                        ):
                            session_id = redaction_data.get("session_id")
                            if session_id and text_to_descrub:
                                with st.spinner(f"De-scrubbing {descrub_choice.lower() if descrub_choice else 'content'}..."):
                                    if st.session_state.get("demo_mode"):
                                        if descrub_type == "original":
                                            demo_text = "[DEMO] This would be the original input text before redaction"
                                        else:
                                            demo_text = "[DEMO] This would be the original Gemini response if it received the unredacted text: 'Here is the original sensitive information that was hidden from the redacted version.'"
                                        
                                        descrub_result = {
                                            "success": True,
                                            "original_text": demo_text,
                                            "session_id": session_id,
                                            "descrub_type": descrub_type,
                                            "descrub_choice": descrub_choice
                                        }
                                    else:
                                        if descrub_type == "original":
                                            # Use the descrub API for original text
                                            success, descrub_result = call_descrub_api(
                                                session_id, text_to_descrub
                                            )
                                            if success:
                                                descrub_result["descrub_type"] = descrub_type
                                                descrub_result["descrub_choice"] = descrub_choice
                                            else:
                                                st.error(
                                                    f"❌ Descrub Error: {descrub_result.get('error', 'Unknown error')}"
                                                )
                                                descrub_result = None
                                        else:  # descrub_type == "gemini"
                                            # For Gemini response, we need to call Gemini with the original text
                                            # First get the original text via descrub API
                                            original_redacted_text = redaction_data.get("redacted_text", "")
                                            success, original_descrub = call_descrub_api(
                                                session_id, original_redacted_text
                                            )
                                            
                                            if success:
                                                original_text = original_descrub.get("original_text", "")
                                                if original_text:
                                                    # Now call Gemini with the original unredacted text
                                                    success, gemini_original_result = call_predict_api(original_text)
                                                    if success:
                                                        descrub_result = {
                                                            "success": True,
                                                            "original_text": gemini_original_result.get("prediction", ""),
                                                            "session_id": session_id,
                                                            "descrub_type": descrub_type,
                                                            "descrub_choice": descrub_choice,
                                                            "model_used": gemini_original_result.get("model_used", "gemini-pro")
                                                        }
                                                    else:
                                                        st.error(
                                                            f"❌ Error getting original Gemini response: {gemini_original_result.get('error', 'Unknown error')}"
                                                        )
                                                        descrub_result = None
                                                else:
                                                    st.error("❌ Could not retrieve original text for Gemini re-processing")
                                                    descrub_result = None
                                            else:
                                                st.error(
                                                    f"❌ Error retrieving original text: {original_descrub.get('error', 'Unknown error')}"
                                                )
                                                descrub_result = None

                                    if descrub_result:
                                        st.session_state.descrub_result = descrub_result
                                        st.success(
                                            f"✅ {descrub_choice} de-scrubbed successfully! See results in the right panel."
                                        )
                            else:
                                st.error("No session ID or content available for de-scrubbing.")
                    else:
                        st.info("No content available for de-scrubbing. Please redact some text first.")
                else:
                    st.info("🔒 De-scrubbing is only available for admin users.")

            # Clear workflow button
            if hasattr(st.session_state, "redaction_result"):
                if st.button("🗑️ Clear Workflow", key="clear_workflow_btn"):
                    for key in [
                        "redaction_result",
                        "prediction_result",
                        "descrub_result",
                    ]:
                        if hasattr(st.session_state, key):
                            delattr(st.session_state, key)
                    st.rerun()

        with tab2:
            st.subheader("File Upload & Processing")

            # File types supported by the API
            file_types = ["pdf", "docx", "txt", "html", "png", "jpg", "jpeg"]
            help_text = "Supported: PDF, DOCX, TXT, HTML, PNG/JPG (with OCR)"

            uploaded_file = st.file_uploader(
                "Choose a file",
                type=file_types,
                help=help_text,
            )

            if uploaded_file is not None:
                st.success(f"File uploaded: {uploaded_file.name}")

                # File info
                file_size = len(uploaded_file.getvalue())
                st.caption(f"📄 Size: {file_size:,} bytes")

                # Sensitivity level selector for files
                file_sensitivity_level = st.selectbox(
                    "Processing Level:",
                    options=["C1", "C2", "C3", "C4"],
                    format_func=lambda x: f"Level {x}",
                    help="Higher levels redact more sensitive information",
                    key="file_sensitivity",
                )

                # Add submit button for file processing
                if st.button("🔍 Process File", key="process_file_btn", type="primary"):
                    with st.spinner("Processing file through secure API pipeline..."):
                        if st.session_state.get("demo_mode"):
                            st.info(
                                "🚧 **Demo Mode**: File processing would use the API"
                            )
                        else:
                            # Use real API
                            file_bytes = uploaded_file.getvalue()
                            success, scrubbed_bytes, scrubbed_filename, content_type = (
                                call_scrub_file_download_api(
                                    file_bytes,
                                    uploaded_file.name,
                                    file_sensitivity_level,
                                )
                            )

                            if success:
                                st.success(f"✅ File processed successfully!")
                                st.download_button(
                                    label=f"📥 Download {scrubbed_filename}",
                                    data=scrubbed_bytes,
                                    file_name=scrubbed_filename,
                                    mime=content_type,
                                    help="Download the processed file with sensitive information redacted",
                                )
                            else:
                                st.error(f"❌ API Error: {content_type}")

        # Admin Logs Tab (only shown for admins)
        if is_admin and tab3 is not None:
            with tab3:
                st.subheader("📊 System Audit Logs")
                st.caption("Enhanced audit logging with banking compliance features")
                
                # Controls for log filtering
                col_a, col_b = st.columns(2)
                
                with col_a:
                    log_limit = st.selectbox(
                        "Number of records:",
                        options=[10, 25, 50, 100],
                        index=1  # Default to 25
                    )
                
                with col_b:
                    if st.button("🔄 Refresh Logs", key="refresh_logs"):
                        # Clear any cached log data
                        if hasattr(st.session_state, "logs_data"):
                            del st.session_state.logs_data
                
                # Fetch and display logs
                if st.button("📊 Load Audit Logs", key="load_logs", type="primary") or hasattr(st.session_state, "logs_data"):
                    
                    if not hasattr(st.session_state, "logs_data"):
                        with st.spinner("Fetching audit logs from MongoDB..."):
                            if st.session_state.get("demo_mode"):
                                # Demo data
                                import datetime
                                st.session_state.logs_data = {
                                    "success": True,
                                    "logs": [
                                        {
                                            "session_id": "demo-001",
                                            "user_id": "admin@ing.com",
                                            "action": "text_redaction",
                                            "timestamp": datetime.datetime.now().isoformat(),
                                            "details": {
                                                "audit_context": {
                                                    "ip_address": "192.168.1.100",
                                                    "user_agent": "Mozilla/5.0...",
                                                    "risk_level": "low"
                                                },
                                                "customer_data_operation": {
                                                    "operation_type": "data_redaction",
                                                    "data_categories": ["personal_info", "financial_data"],
                                                    "risk_classification": "medium",
                                                    "requires_approval": False
                                                }
                                            }
                                        },
                                        {
                                            "session_id": "demo-002", 
                                            "user_id": "user@ing.com",
                                            "action": "file_upload",
                                            "timestamp": (datetime.datetime.now() - datetime.timedelta(hours=1)).isoformat(),
                                            "details": {
                                                "audit_context": {
                                                    "ip_address": "192.168.1.101",
                                                    "user_agent": "Mozilla/5.0...",
                                                    "risk_level": "medium"
                                                },
                                                "customer_data_operation": {
                                                    "operation_type": "file_processing",
                                                    "data_categories": ["customer_data"],
                                                    "risk_classification": "high",
                                                    "requires_approval": True
                                                }
                                            }
                                        }
                                    ]
                                }
                            else:
                                success, logs_data = call_logs_api(log_limit)
                                if success:
                                    st.session_state.logs_data = logs_data
                                else:
                                    st.error(f"❌ Failed to load logs: {logs_data.get('error', 'Unknown error')}")
                                    st.session_state.logs_data = {"success": False}
                    
                    # Display logs if available
                    if hasattr(st.session_state, "logs_data") and st.session_state.logs_data.get("success"):
                        logs = st.session_state.logs_data.get("logs", [])
                        
                        if logs and isinstance(logs, list):
                            st.success(f"✅ Loaded {len(logs)} audit records")
                            
                            # Summary metrics
                            st.markdown("### 📈 Summary Metrics")
                            
                            col1_metrics, col2_metrics, col3_metrics = st.columns(3)
                            
                            with col1_metrics:
                                total_sessions = len(set(log.get("session_id", "") for log in logs))
                                st.metric("Unique Sessions", total_sessions)
                            
                            with col2_metrics:
                                high_risk_ops = len([log for log in logs 
                                                   if log.get("details", {}).get("customer_data_operation", {}).get("risk_classification") == "high"])
                                st.metric("High Risk Operations", high_risk_ops)
                            
                            with col3_metrics:
                                unique_users = len(set(log.get("user_id", "") for log in logs))
                                st.metric("Active Users", unique_users)
                            
                            # Detailed log view
                            st.markdown("### 📋 Detailed Audit Log")
                            
                            for i, log in enumerate(logs):
                                with st.expander(f"🔍 {log.get('action', 'Unknown')} - {log.get('session_id', 'N/A')[:8]}..."):
                                    col_log1, col_log2 = st.columns(2)
                                    
                                    with col_log1:
                                        st.write("**Basic Info:**")
                                        st.write(f"- **User:** {log.get('user_id', 'N/A')}")
                                        st.write(f"- **Action:** {log.get('action', 'N/A')}")
                                        st.write(f"- **Session:** {log.get('session_id', 'N/A')}")
                                        st.write(f"- **Time:** {log.get('timestamp', 'N/A')}")
                                    
                                    with col_log2:
                                        audit_context = log.get("details", {}).get("audit_context", {})
                                        customer_op = log.get("details", {}).get("customer_data_operation", {})
                                        
                                        st.write("**Security Context:**")
                                        st.write(f"- **IP:** {audit_context.get('ip_address', 'N/A')}")
                                        st.write(f"- **Risk Level:** {audit_context.get('risk_level', 'N/A')}")
                                        
                                        st.write("**Data Operation:**")
                                        st.write(f"- **Type:** {customer_op.get('operation_type', 'N/A')}")
                                        st.write(f"- **Risk Class:** {customer_op.get('risk_classification', 'N/A')}")
                                        
                                        # Risk indicator
                                        risk = customer_op.get('risk_classification', 'unknown')
                                        if risk == 'high':
                                            st.error("🚨 High Risk Operation")
                                        elif risk == 'medium':
                                            st.warning("⚠️ Medium Risk Operation")
                                        else:
                                            st.success("✅ Low Risk Operation")
                        else:
                            st.info("No audit logs found.")

        st.markdown("</div>", unsafe_allow_html=True)  # Close ing-container for col1

    with col2:
        st.markdown('<div class="ing-container">', unsafe_allow_html=True)
        st.header("🎯 Workflow Results")

        # Check if we have any workflow results
        has_redaction = hasattr(st.session_state, "redaction_result")
        has_prediction = hasattr(st.session_state, "prediction_result")
        has_descrub = hasattr(st.session_state, "descrub_result")

        if has_redaction or has_prediction or has_descrub:
            # Step 1: Redaction Results
            if has_redaction:
                redaction_data = st.session_state.redaction_result

                st.subheader("📊 Step 1: Redaction Results")

                # Show statistics
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    original_text = redaction_data.get("original_text", "")
                    st.metric("Original Length", f"{len(original_text):,} chars")
                with col_b:
                    redacted_text = redaction_data.get("redacted_text", "")
                    st.metric("Redacted Length", f"{len(redacted_text):,} chars")
                with col_c:
                    st.metric("Items Redacted", redaction_data.get("total_redacted", 0))

                # Show session info
                session_id = redaction_data.get("session_id", "N/A")
                st.info(f"🔑 **Session ID**: `{session_id}`")

                # Show detections if any
                detections = redaction_data.get("detections", [])
                if detections:
                    st.warning("⚠️ Sensitive information detected and redacted!")

                    with st.expander("🔍 View Detected Items", expanded=False):
                        detection_summary = {}
                        for detection in detections:
                            det_type = detection.get("type", "Unknown")
                            detection_summary[det_type] = (
                                detection_summary.get(det_type, 0) + 1
                            )

                        for det_type, count in detection_summary.items():
                            st.write(f"• **{det_type}**: {count} instance(s)")

                # Show redacted content (read-only)
                st.markdown("**✅ Redacted Content (Read-Only):**")
                st.text_area(
                    "Redacted Text:",
                    value=redacted_text,
                    height=150,
                    disabled=True,
                    key="redacted_display",
                )

                st.markdown("---")

            # Step 2: Gemini Results
            if has_prediction:
                prediction_data = st.session_state.prediction_result

                st.subheader("🧠 Step 2: Gemini Response")

                if prediction_data.get("success"):
                    st.success("✅ Gemini analysis complete!")

                    gemini_response = prediction_data.get("prediction", "No response generated")
                    st.markdown("**🤖 Gemini Response to Redacted Text:**")
                    st.text_area(
                        "Gemini Response:",
                        value=gemini_response,
                        height=150,
                        disabled=True,
                        key="gemini_response_display",
                    )

                    # Additional info if available
                    model_used = prediction_data.get("model_used", "Unknown")
                    st.caption(f"Model: {model_used}")
                else:
                    st.error("Gemini analysis failed")

                st.markdown("---")

            # Step 3: Descrub Results (Admin only)
            if has_descrub:
                user_info = st.session_state.get(USER_INFO_KEY, {})
                is_admin = user_info.get("is_admin", False)

                if is_admin:
                    descrub_data = st.session_state.descrub_result

                    # Get the type of content that was descrubbed
                    descrub_choice = descrub_data.get("descrub_choice", "Content")
                    descrub_type = descrub_data.get("descrub_type", "unknown")

                    st.subheader(f"🔓 Step 3: De-scrub Results - {descrub_choice} (Admin)")

                    if descrub_data.get("success"):
                        if descrub_type == "original":
                            st.success("✅ Original input text successfully restored!")
                            content_label = "🔓 Original Input Text (Restored)"
                            download_filename = "original_input_text.txt"
                        elif descrub_type == "gemini":
                            st.success("✅ Original Gemini response successfully restored!")
                            content_label = "🤖 Original Gemini Response (Restored)"
                            download_filename = "original_gemini_response.txt"
                        else:
                            st.success("✅ Content successfully restored!")
                            content_label = "🔓 Original Content (Restored)"
                            download_filename = "restored_content.txt"

                        original_text = descrub_data.get("original_text", "")
                        st.markdown(f"**{content_label}:**")
                        st.text_area(
                            f"Restored {descrub_choice}:",
                            value=original_text,
                            height=150,
                            disabled=True,
                            key="original_display",
                        )

                        # Download restored text
                        st.download_button(
                            label=f"📥 Download {descrub_choice}",
                            data=original_text,
                            file_name=download_filename,
                            mime="text/plain",
                            help=f"Download the restored {descrub_choice.lower()}",
                        )
                    else:
                        st.error("Failed to restore original text")

            # Workflow Progress Indicator
            st.markdown("---")
            st.subheader("🔄 Workflow Progress")

            progress_steps = [
                ("1️⃣ Redaction", has_redaction),
                ("2️⃣ Prediction", has_prediction),
                ("3️⃣ De-scrub (Admin)", has_descrub),
            ]

            for step_name, completed in progress_steps:
                if completed:
                    st.success(f"✅ {step_name}")
                else:
                    st.info(f"⏳ {step_name}")

        else:
            st.info(
                "👆 Start the workflow by entering text and clicking 'Step 1: Redact Text'"
            )

            # Show workflow explanation
            st.markdown("### 🔄 Redaction Workflow")
            st.markdown(
                """
            **Step-by-step process:**
            
            1. **🔍 Redaction**: Enter text and redact sensitive information
               - Uses `/redact` API endpoint
               - Creates a session ID for tracking
               - Shows redacted text (read-only)
            
            2. **🧠 Prediction**: Analyze original text classification
               - Uses `/predict` API endpoint  
               - Simulates internal ML processing
               - Shows confidence and explanation
            
            3. **🔓 De-scrub** (Admin only): Restore original text
               - Uses `/de-scrub` API endpoint with session ID
               - Only available for admin users
               - Shows original unredacted text
            
            **Features:**
            - 🔒 **Session tracking**: Each redaction creates a unique session
            - 👑 **Admin controls**: De-scrubbing requires admin permissions
            - 📊 **Read-only results**: Redacted content cannot be modified
            - 🔄 **Complete workflow**: Demonstrates full pipeline
            """
            )

        st.markdown("</div>", unsafe_allow_html=True)  # Close ing-container for col2

    # ING Brand Footer
    st.markdown(
        """
    <div class="footer">
        <div class="ing-accent" style="margin: 0; height: 2px;"></div>
        🔍 ING Prompt Scrubber | Powered by ING Bank Technology | Secure Document Processing
    </div>
    """,
        unsafe_allow_html=True,
    )


def main():
    """Main application entry point with API integration."""
    # Initialize session state
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    # Route to appropriate page
    if st.session_state.logged_in:
        main_app()
    else:
        login_page()


if __name__ == "__main__":
    main()
