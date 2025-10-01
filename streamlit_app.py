import streamlit as st
from src.prompt_scrubber import PromptScrubber
from src.file_handler.docx_to_txt import DOCXToTextConverter
import io


def login_page():
    """Display login page with role selection."""
    st.set_page_config(
        page_title="Prompt Scrubber - Login", page_icon="🔐", layout="centered"
    )

    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.title("🔐 Login")
        st.markdown("---")

        with st.container():

            st.subheader("Welcome")
            st.markdown("Please select your role to continue:")

            # Role selection dropdown
            def format_role(role_option):
                role_mapping = {
                    "": "-- Select Role --",
                    "admin": "👑 Admin",
                    "user": "👤 User",
                }
                return role_mapping.get(role_option, str(role_option))

            role = st.selectbox(
                "Select your role:",
                options=["", "admin", "user"],
                format_func=format_role,
                key="role_selection",
            )

            # Login button
            if st.button("🚀 Login", type="primary", use_container_width=True):
                if role:
                    # Store login state in session
                    st.session_state.logged_in = True
                    st.session_state.user_role = role
                    st.success(f"Welcome, {st.session_state.user_role}!")
                    st.balloons()
                    st.rerun()
                else:
                    st.error("Please select a role to continue.")

            st.markdown("</div>", unsafe_allow_html=True)


def logout():
    """Handle user logout."""
    for key in ["logged_in", "user_role", "username"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()


def main_app():
    """Main application after login."""
    st.set_page_config(page_title="Prompt Scrubber App", page_icon="🔍", layout="wide")

    # Header with user info and logout
    col1, col2 = st.columns([3, 1])

    with col2:
        if st.button("🚪 Logout", key="logout_btn"):
            logout()

    @st.cache_resource
    def get_scrubber():
        return PromptScrubber()

    scrubber = get_scrubber()

    # Create two columns for layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📝 Input Options")

        # Create tabs for different input methods
        tab1, tab2 = st.tabs(["Text Input", "File Upload"])

        text_content = None

        with tab1:
            st.subheader("Direct Text Input")
            text_input = st.text_area(
                "Enter your prompt here:",
                placeholder="Type or paste your text here...",
                height=200,
            )

            # Add submit button for text input
            text_submitted = st.button(
                "Scrub Text", key="scrub_text_btn", type="primary"
            )

            if text_input and text_submitted:
                text_content = text_input

        with tab2:
            st.subheader("File Upload")
            file_types = ["pdf", "docx", "txt", "html", "xml", "json"]
            help_text = (
                "Supported formats: PDF, DOCX, TXT, HTML, XML, JSON (Admin access)"
            )

            uploaded_file = st.file_uploader(
                "Choose a file",
                type=file_types,
                help=help_text,
            )

            if uploaded_file is not None:
                st.success(f"File uploaded: {uploaded_file.name}")

                # Add submit button for file processing
                file_submitted = st.button(
                    "🔍 Analyze File", key="analyze_file_btn", type="primary"
                )

                if file_submitted:
                    with st.spinner("Extracting text from file..."):
                        try:
                            # Process based on file type
                            if uploaded_file.type == "text/plain":
                                text_content = str(uploaded_file.read(), "utf-8")
                            elif (
                                uploaded_file.type
                                == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                            ):

                                converter = DOCXToTextConverter()

                                # Save uploaded file temporarily with unique name
                                import uuid

                                unique_id = str(uuid.uuid4())[:8]
                                temp_file_path = (
                                    f"temp_{unique_id}_{uploaded_file.name}"
                                )
                                temp_txt_path = temp_file_path.replace(".docx", ".txt")

                                try:
                                    with open(temp_file_path, "wb") as f:
                                        f.write(uploaded_file.getbuffer())

                                    # Convert DOCX to text
                                    success = converter.convert_docx_to_txt(
                                        temp_file_path, temp_txt_path
                                    )

                                    if success:
                                        with open(
                                            temp_txt_path, "r", encoding="utf-8"
                                        ) as f:
                                            text_content = f.read()

                                        st.success(
                                            "✅ DOCX file processed successfully!"
                                        )

                                    else:
                                        st.error("❌ Failed to process DOCX file")

                                finally:
                                    # Clean up temporary files
                                    import os

                                    for temp_file in [temp_file_path, temp_txt_path]:
                                        if os.path.exists(temp_file):
                                            try:
                                                os.remove(temp_file)
                                            except:
                                                pass  # Ignore cleanup errors

                            else:
                                st.warning(
                                    f"File type '{uploaded_file.type}' not fully supported yet. "
                                    "Please use TXT, DOCX, or HTML files."
                                )

                        except Exception as e:
                            st.error(f"Error processing file: {str(e)}")
                            st.info("Please try a different file or contact support.")

    with col2:
        st.header("Scrubbing Results")

        if text_content:
            st.header("Scrubbing Results")
            st.subheader("Original Content Preview")
            with st.expander("Click to view full content"):
                st.text_area("Full Content:", text_content, height=200, disabled=True)

            # Show first 200 characters as preview
            preview = (
                text_content[:200] + "..." if len(text_content) > 200 else text_content
            )
            st.text(f"Preview: {preview}")

            st.subheader("Scrubbing Analysis")

            with st.spinner("Analyzing content for classified data..."):
                # Check for matches
                matches = scrubber.scrub(text_content)

                # Get scrubbed prompt
                scrubbed_prompt = scrubber.scrub_prompt(text_content)

            if matches:
                st.error("⚠️ Classified content detected!")

                with st.expander("View detected matches", expanded=True):
                    for filename, found_values in matches.items():
                        st.write(f"**File: {filename}**")
                        for value in found_values:
                            st.write(f"  • {value}")

                st.subheader("Scrubbed Content")
                st.success("✅ Content has been scrubbed")

                st.code(scrubbed_prompt, language=None)

            else:
                st.success("✅ No classified content detected!")
                st.info("The content is safe to use as-is.")

                # Admin-only clean content details
                if st.session_state.user_role == "admin":
                    st.markdown(
                        "**🔧 Admin Info:** Content passed all classification checks."
                    )

    # Footer with role-specific information
    st.markdown("---")
    if st.session_state.user_role == "admin":
        st.markdown("*🔧 Admin Dashboard • Full Access Mode *")
    else:
        st.markdown(
            "*👤 User Dashboard • Standard Access • Contact your administrator for advanced features*"
        )


def main():
    """Main application entry point."""
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
