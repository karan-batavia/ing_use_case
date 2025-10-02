import streamlit as st
from src.prompt_scrubber import PromptScrubber
from src.file_handler.docx_to_txt import DOCXToTextConverter
from src.file_handler.html_to_txt import HTMLToTextConverter
from src.file_handler.read_pdf_file import PDFToTextConverter
from src.file_handler.read_png_file import ImageToTextConverter
from src.file_handler.write_docx_file import DOCXWriter
from src.file_handler.write_pdf_file import PDFWriter
from src.file_handler.write_html_file import HTMLWriter
from src.file_handler.write_txt_file import TXTWriter
from src.mongodb_service import (
    get_mongodb_service,
    ensure_session_tracking,
    log_app_interaction,
)
import io
import uuid
import os


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
                    user_id = (
                        f"{role}_{uuid.uuid4().hex[:8]}"  # Generate unique user ID
                    )
                    st.session_state.logged_in = True
                    st.session_state.user_role = role
                    st.session_state.user_id = user_id

                    # Initialize MongoDB session tracking
                    mongodb_service = get_mongodb_service()
                    session_id = mongodb_service.create_session(user_id, role)
                    st.session_state.session_id = session_id
                    mongodb_service.log_login(user_id, role)

                    # Log the login interaction
                    log_app_interaction("login", {"user_role": role})

                    st.success(f"Welcome, {st.session_state.user_role}!")
                    st.balloons()
                    st.rerun()
                else:
                    st.error("Please select a role to continue.")

            st.markdown("</div>", unsafe_allow_html=True)


def logout():
    """Handle user logout."""
    # End MongoDB session before clearing session state
    if "session_id" in st.session_state:
        mongodb_service = get_mongodb_service()
        log_app_interaction("logout")
        mongodb_service.end_session(st.session_state.session_id)

    # Clear session state
    for key in ["logged_in", "user_role", "username", "user_id", "session_id"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()


def create_download_button(
    scrubbed_text: str, original_filename: str, file_extension: str
):
    """Create a download button for scrubbed content in the original file format."""
    try:
        file_base_name = (
            original_filename.rsplit(".", 1)[0]
            if "." in original_filename
            else original_filename
        )
        scrubbed_filename = f"{file_base_name}_scrubbed{file_extension}"

        if file_extension.lower() == ".docx":
            writer = DOCXWriter()
            file_buffer = writer.create_docx_buffer(scrubbed_text)
            mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

        elif file_extension.lower() == ".pdf":
            writer = PDFWriter()
            file_buffer = writer.create_pdf_buffer(scrubbed_text)
            mime_type = "application/pdf"

        elif file_extension.lower() == ".html":
            writer = HTMLWriter()
            file_buffer = writer.create_html_buffer(scrubbed_text)
            mime_type = "text/html"

        elif file_extension.lower() == ".txt":
            writer = TXTWriter()
            file_buffer = writer.create_txt_buffer(scrubbed_text)
            mime_type = "text/plain"

        else:
            # For PNG and other image files, create a TXT file with the scrubbed OCR content
            writer = TXTWriter()
            file_buffer = writer.create_txt_buffer(scrubbed_text)
            scrubbed_filename = f"{file_base_name}_scrubbed_ocr.txt"
            mime_type = "text/plain"

        download_clicked = st.download_button(
            label=f"📥 Download {scrubbed_filename}",
            data=file_buffer.getvalue(),
            file_name=scrubbed_filename,
            mime=mime_type,
            help=f"Download the scrubbed content as {scrubbed_filename}",
        )

        # Log download interaction when button is clicked
        if download_clicked:
            log_app_interaction(
                "file_download",
                {
                    "filename": scrubbed_filename,
                    "file_type": file_extension,
                    "file_size": len(file_buffer.getvalue()),
                },
            )

        return download_clicked

    except Exception as e:
        st.error(f"Error creating download: {str(e)}")
        return False


def main_app():
    """Main application after login."""
    st.set_page_config(page_title="Prompt Scrubber App", page_icon="🔍", layout="wide")

    # Ensure session tracking is set up
    ensure_session_tracking()

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
                # Log text input interaction
                log_app_interaction("text_input", {"input_length": len(text_input)})

                # Clear file session state when using text input
                if hasattr(st.session_state, "uploaded_filename"):
                    del st.session_state.uploaded_filename
                if hasattr(st.session_state, "file_extension"):
                    del st.session_state.file_extension

        with tab2:
            st.subheader("File Upload")

            file_types = ["pdf", "docx", "txt", "html", "png"]
            help_text = "Supported formats: PDF, DOCX, TXT, HTML, PNG (Admin access)"

            uploaded_file = st.file_uploader(
                "Choose a file",
                type=file_types,
                help=help_text,
            )

            if uploaded_file is not None:
                st.success(f"File uploaded: {uploaded_file.name}")

                # Log file upload
                file_size = (
                    len(uploaded_file.getbuffer())
                    if hasattr(uploaded_file, "getbuffer")
                    else None
                )
                log_app_interaction(
                    "file_upload",
                    {
                        "filename": uploaded_file.name,
                        "file_type": uploaded_file.type,
                        "file_size": file_size,
                    },
                )

                # Store uploaded file info in session state
                st.session_state.uploaded_filename = uploaded_file.name
                file_extension = (
                    "." + uploaded_file.name.lower().split(".")[-1]
                    if "." in uploaded_file.name
                    else ""
                )
                st.session_state.file_extension = file_extension

                # Add submit button for file processing
                file_submitted = st.button(
                    "🔍 Analyze File", key="analyze_file_btn", type="primary"
                )

                if file_submitted:
                    with st.spinner("Extracting text from file..."):
                        try:
                            text_content = None

                            # Helper function to create unique temporary file names
                            def get_temp_filename(original_name):
                                unique_id = str(uuid.uuid4())[:8]
                                return f"temp_{unique_id}_{original_name}"

                            # Helper function to clean up temporary files
                            def cleanup_temp_files(*file_paths):
                                for file_path in file_paths:
                                    if os.path.exists(file_path):
                                        try:
                                            os.remove(file_path)
                                        except:
                                            pass  # Ignore cleanup errors

                            # Process based on file type and extension
                            file_extension = uploaded_file.name.lower().split(".")[-1]

                            if (
                                uploaded_file.type == "text/plain"
                                or file_extension == "txt"
                            ):
                                # Plain text files
                                text_content = str(uploaded_file.read(), "utf-8")
                                st.success("✅ Text file processed successfully!")

                            elif (
                                uploaded_file.type
                                == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                                or file_extension == "docx"
                            ):
                                # DOCX files
                                converter = DOCXToTextConverter()
                                temp_file_path = get_temp_filename(uploaded_file.name)
                                temp_txt_path = temp_file_path.replace(".docx", ".txt")

                                try:
                                    with open(temp_file_path, "wb") as f:
                                        f.write(uploaded_file.getbuffer())

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
                                    cleanup_temp_files(temp_file_path, temp_txt_path)

                            elif (
                                uploaded_file.type == "application/pdf"
                                or file_extension == "pdf"
                            ):
                                # PDF files
                                converter = PDFToTextConverter()
                                temp_file_path = get_temp_filename(uploaded_file.name)

                                try:
                                    with open(temp_file_path, "wb") as f:
                                        f.write(uploaded_file.getbuffer())

                                    text_content = converter.extract_text_from_pdf(
                                        temp_file_path
                                    )

                                    if text_content:
                                        st.success(
                                            "✅ PDF file processed successfully!"
                                        )
                                    else:
                                        st.error(
                                            "❌ Failed to extract text from PDF file"
                                        )

                                finally:
                                    cleanup_temp_files(temp_file_path)

                            elif (
                                uploaded_file.type == "text/html"
                                or file_extension in ["html", "htm"]
                            ):
                                # HTML files
                                converter = HTMLToTextConverter(
                                    clean_whitespace=True,
                                    remove_empty_lines=True,
                                    preserve_links=False,
                                )
                                temp_file_path = get_temp_filename(uploaded_file.name)
                                temp_txt_path = temp_file_path.replace(
                                    ".html", ".txt"
                                ).replace(".htm", ".txt")

                                try:
                                    with open(temp_file_path, "wb") as f:
                                        f.write(uploaded_file.getbuffer())

                                    success = converter.convert_html_to_txt(
                                        temp_file_path, temp_txt_path
                                    )

                                    if success:
                                        with open(
                                            temp_txt_path, "r", encoding="utf-8"
                                        ) as f:
                                            text_content = f.read()
                                        st.success(
                                            "✅ HTML file processed successfully!"
                                        )
                                    else:
                                        st.error("❌ Failed to process HTML file")

                                finally:
                                    cleanup_temp_files(temp_file_path, temp_txt_path)

                            elif uploaded_file.type.startswith(
                                "image/"
                            ) or file_extension in [
                                "png",
                                "jpg",
                                "jpeg",
                                "tiff",
                                "bmp",
                            ]:
                                # Image files (OCR)
                                converter = ImageToTextConverter(
                                    languages="eng+fra+nld",  # English, French, Dutch
                                    preprocess=True,
                                )
                                temp_file_path = get_temp_filename(uploaded_file.name)

                                try:
                                    with open(temp_file_path, "wb") as f:
                                        f.write(uploaded_file.getbuffer())

                                    text_content = converter.extract_text_from_image(
                                        temp_file_path
                                    )

                                    if text_content:
                                        st.success(
                                            "✅ Image file processed successfully using OCR!"
                                        )
                                        st.info(
                                            "📝 Text extracted using OCR (English, French, Dutch)"
                                        )
                                    else:
                                        st.warning(
                                            "⚠️ No text found in the image or OCR failed"
                                        )

                                finally:
                                    cleanup_temp_files(temp_file_path)

                            else:
                                st.warning(
                                    f"File type '{uploaded_file.type}' or extension '{file_extension}' is not supported. "
                                    f"Please use: {', '.join(file_types).upper()} files."
                                )

                        except Exception as e:
                            st.error(f"Error processing file: {str(e)}")
                            st.info("Please try a different file or contact support.")

    with col2:
        st.header("Scrubbing Results")

        if text_content:
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

                # Log scrubbing interaction
                matches_count = (
                    sum(len(found_values) for found_values in matches.values())
                    if matches
                    else 0
                )
                log_app_interaction(
                    "text_scrubbing",
                    {
                        "input_length": len(text_content),
                        "output_length": len(scrubbed_prompt),
                        "matches_found": matches_count,
                        "reduction_percentage": (
                            round(
                                (1 - len(scrubbed_prompt) / len(text_content)) * 100, 2
                            )
                            if len(text_content) > 0
                            else 0
                        ),
                    },
                )

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

                # Add download button for scrubbed content
                st.markdown("---")
                st.subheader("📥 Download Scrubbed File")

                # Check if we have file info from upload
                if hasattr(st.session_state, "uploaded_filename") and hasattr(
                    st.session_state, "file_extension"
                ):
                    if create_download_button(
                        scrubbed_prompt,
                        st.session_state.uploaded_filename,
                        st.session_state.file_extension,
                    ):
                        st.success("Download initiated! 🎉")
                else:
                    # Fallback for text input - provide TXT download
                    writer = TXTWriter()
                    txt_buffer = writer.create_txt_buffer(scrubbed_prompt)
                    st.download_button(
                        label="📥 Download Scrubbed Text (TXT)",
                        data=txt_buffer.getvalue(),
                        file_name="scrubbed_text.txt",
                        mime="text/plain",
                        help="Download the scrubbed content as a text file",
                    )

            else:
                st.success("✅ No classified content detected!")
                st.info("The content is safe to use as-is.")

                # Still provide download option for the original content
                st.markdown("---")
                st.subheader("📥 Download Original File")
                st.info(
                    "Since no classified content was detected, you can download the original content."
                )

                # Check if we have file info from upload
                if hasattr(st.session_state, "uploaded_filename") and hasattr(
                    st.session_state, "file_extension"
                ):
                    if create_download_button(
                        text_content,
                        st.session_state.uploaded_filename,
                        st.session_state.file_extension,
                    ):
                        st.success("Download initiated! 🎉")
                else:
                    # Fallback for text input - provide TXT download
                    writer = TXTWriter()
                    txt_buffer = writer.create_txt_buffer(text_content)
                    st.download_button(
                        label="📥 Download Original Text (TXT)",
                        data=txt_buffer.getvalue(),
                        file_name="original_text.txt",
                        mime="text/plain",
                        help="Download the original content as a text file",
                    )

                # Admin-only clean content details
                if st.session_state.user_role == "admin":
                    st.markdown(
                        "**🔧 Admin Info:** Content passed all classification checks."
                    )

    # Footer with role-specific information
    st.markdown("---")
    if st.session_state.user_role == "admin":
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("*🔧 Admin Dashboard • Full Access Mode *")

        with col2:
            # MongoDB connection status and stats for admin
            mongodb_service = get_mongodb_service()
            if mongodb_service.is_connected():
                st.success("🟢 MongoDB Connected")

                # Show session stats
                if "session_id" in st.session_state:
                    session_stats = mongodb_service.get_session_stats(
                        st.session_state.session_id
                    )
                    if session_stats:
                        st.info(
                            f"Session interactions: {session_stats.get('total_interactions', 0)}"
                        )

                # Show user stats
                if "user_id" in st.session_state:
                    user_stats = mongodb_service.get_user_stats(
                        st.session_state.user_id
                    )
                    if user_stats:
                        st.info(
                            f"Total user sessions: {user_stats.get('total_sessions', 0)}"
                        )
            else:
                st.warning("🔴 MongoDB Disconnected")
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
