from typing import Optional, List
import os
import shutil
from moviepy.video.io.VideoFileClip import VideoFileClip
from pydub import AudioSegment
import whisper
import torch
import flet as ft
import tempfile

def extract_audio_from_video(video_path: str, audio_path: str) -> str:
    """Extract audio from video and save to file."""
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path)
    print("Audio extracted from video:", audio_path)
    return audio_path

def split_audio(input_audio: str, chunk_length_ms: int = 5*60*1000) -> List[str]:
    """Split a long audio file into several chunks. Returns chunk file path list."""
    audio = AudioSegment.from_file(input_audio)
    chunk_paths = []
    for i, start in enumerate(range(0, len(audio), chunk_length_ms)):
        chunk = audio[start:start+chunk_length_ms]
        chunk_path = f"{input_audio}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunk_paths.append(chunk_path)
    return chunk_paths

def transcribe_chunks(model, chunk_paths: List[str], device: str) -> List[dict]:
    """Run whisper transcription for each chunk. Returns combined segments."""
    all_segments = []
    for chunk_path in chunk_paths:
        print(f"Transcribing: {chunk_path}")
        result = model.transcribe(chunk_path, verbose=False)
        if "segments" in result:
            all_segments.extend(result["segments"])
        else:
            # For base/small models without segments info
            all_segments.append({"start":0, "end":0, "text": result.get("text", "")})
    return all_segments

def save_transcript(segments: List[dict], transcript_path: str):
    """Save segments as timestamped transcript to file."""
    os.makedirs(os.path.dirname(transcript_path), exist_ok=True)
    with open(transcript_path, "w", encoding="utf-8") as f:
        for seg in segments:
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            text = seg.get("text", "")
            f.write(f"[{start:.1f} ~ {end:.1f}] {text}\n")

def delete_whisper_cache(cache_path: str):
    """Delete whisper model cache directory."""
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)
        print("Whisper cache deleted:", cache_path)

def main_gui(page: ft.Page):
    # color
    page.title = "Interview Analyzer (Whisper GUI)"
    page.bgcolor = ft.Colors.BLUE_GREY_900
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER

    show_transcript = ft.Ref()    
    show_transcript.current = False
    file_picker = ft.FilePicker()
    upload_button = ft.ElevatedButton("Select Audio/Video File", on_click=lambda _: file_picker.pick_files())
    status_text = ft.Text("")
    progress_bar = ft.ProgressBar(width=400, height=10, value=0)
    result_box = ft.TextField(label="Transcript", multiline=True, min_lines=10, max_lines=30, read_only=True, expand=True, visible=show_transcript.current)
    save_button = ft.ElevatedButton("Download TXT", visible=False)
    model_dropdown = ft.Dropdown(
        label="Whisper Model", options=[ft.dropdown.Option("base"), ft.dropdown.Option("small"),
                                       ft.dropdown.Option("medium"), ft.dropdown.Option("large")],
                                       value="large",   
                                       label_style=ft.TextStyle(color=ft.Colors.WHITE)                                    
                                       
    )
    chunk_input = ft.TextField(label="Chunk Length (minutes)", value="5", width=160, label_style=ft.TextStyle(color=ft.Colors.WHITE))
    file_save_picker = ft.FilePicker()

    cache_confirm_dialog = ft.AlertDialog(
        modal=True,
        title=ft.Text("Delete Whisper Cache?"),
        content=ft.Text("This will remove the downloaded Whisper model files from your disk. Are you sure?"),
        actions=[
            ft.TextButton("Cancel", on_click=lambda e: close_dialog()),
            ft.TextButton("Confirm", style=ft.ButtonStyle(bgcolor=ft.Colors.RED_400, color=ft.Colors.WHITE), on_click=lambda e: do_delete_cache()),
        ],
        actions_alignment=ft.MainAxisAlignment.END,
    )

    def close_dialog():
        cache_confirm_dialog.open = False
        page.update()

    def show_cache_confirm_dialog(e):
        page.dialog = cache_confirm_dialog
        cache_confirm_dialog.open = True
        page.update()
    
    def do_delete_cache():
        cache_path = os.path.expanduser("~/.cache/whisper")
        try:
            delete_whisper_cache(cache_path)
            status_text.value = "Whisper cache deleted."
        except Exception as ex:
            status_text.value = f"Delete Failed: {ex}"
        cache_confirm_dialog.open = False
        page.update()

    delete_cache_button = ft.ElevatedButton(
        "Delete Whisper Cache",
        on_click=show_cache_confirm_dialog,
        style=ft.ButtonStyle(bgcolor=ft.Colors.RED_400, color=ft.Colors.WHITE)
    )

    def toggle_transcript_visibility(e):
        show_transcript.current = not show_transcript.current
        result_box.visible = show_transcript.current
        page.update()
        toggle_button.text = "Show Transcript" if not show_transcript.current else "Hide Transcript"
        page.update()

    toggle_button = ft.ElevatedButton("Hide Transcript", on_click=toggle_transcript_visibility)

    # GUI Components
    app_box = ft.Container(
        content=ft.Column(
            [
                ft.Text("Interview Analyzer", size=20, weight="bold", color=ft.Colors.WHITE),
                ft.Row([upload_button, model_dropdown, chunk_input], alignment=ft.MainAxisAlignment.CENTER),
                progress_bar,
                status_text,
                toggle_button,
                result_box,
                save_button,
                delete_cache_button
            ],
            spacing=16,
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,            
        ),
        width=600,
        padding=30,
        bgcolor=ft.Colors.BLUE_GREY_800,
        border_radius=20,
        shadow=ft.BoxShadow(blur_radius=18, color=ft.Colors.BLACK38)
    )

    # Transcription Logic
    def process_file(e: Optional[ft.FilePickerResultEvent]=None):
        if not file_picker.result or not file_picker.result.files:
            status_text.value = "No File Selected"
            page.update()
            return
        
        file_path = file_picker.result.files[0].path
        file_ext = os.path.splitext(file_path)[-1].lower()
        model_name = model_dropdown.value
        chunk_length_min = int(chunk_input.value)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        status_text.value = f"Loading Whiper Model ({model_name}) On {device}"
        progress_bar.value = 0.1
        page.update()

        # Prepare Audio
        if file_ext in [".mp3", ".wav", ".m4a"]:
            audio_path = file_path
        elif file_ext in [".mp4", ".avi", ".mov"]:
            audio_path = os.path.join(tempfile.gettempdir(), "temp_extracted.wav")
            status_text.value = "Extraction Audio From Video..."
            page.update()
            extract_audio_from_video(file_path, audio_path)
        else:
            status_text.value = "Unsupported File Type!"
            page.update()
            return
        
        # Split Audio
        status_text.value = f"Splitting Audio ({chunk_length_min} min Per Chunk)..."
        page.update()
        chunk_paths = split_audio(audio_path, chunk_length_ms=chunk_length_min*60*1000)

        # Whisper Transcription
        status_text.value = f"Transcribing {len(chunk_paths)} Chunk(s) With Whisper..."
        progress_bar.value = 0.2
        page.update()
        model = whisper.load_model(model_name, device=device)
        segments = []
        for idx, chunk_path in enumerate(chunk_paths):
            status_text.value = f"Transcribing Chunk {idx+1}/{len(chunk_paths)}..."
            progress_bar.value = 0.2 + 0.7 * (idx+1) / len(chunk_paths)
            page.update()
            segments += transcribe_chunks(model, [chunk_path], device)
        progress_bar.value = 1.0
        page.update()

        # Combine Transcript
        transcript_lines = [
            f"[{seg.get('start', 0):.1f} ~ {seg.get('end', 0):.1f}] {seg.get('text', '')}"
            for seg in segments
        ]
        transcript_text = "\n".join(transcript_lines)
        result_box.value = transcript_text
        save_button.visible = True
        status_text.value = f"Transcription Complete ({len(transcript_lines)} Lines)."
        page.update()

        # Clean up temp chunks
        for chunk in chunk_paths:
            try:
                os.remove(chunk)
            except Exception:
                pass
        if file_ext not in [".mp3", ".wav", ".m4a"]:
            try:
                os.remove(audio_path)
            except Exception:
                pass

    
    # Save txt handler
    def save_transcript_file(e):
        file_save_picker.save_file("transcript.txt")

    def file_save_result(e: ft.FilePickerResultEvent):
        if file_save_picker.result and file_save_picker.result.path:
            with open(file_save_picker.result.path, "w", encoding="utf-8") as f:
                f.write(result_box.value)
            status_text.value = f"Transcript Saved: {file_save_picker.result.path}"
            page.update()

    # Events
    file_picker.on_result = process_file
    file_save_picker.on_result = file_save_result
    save_button.on_click = save_transcript_file

    # Add Controls
    page.add(app_box, file_picker, file_save_picker,cache_confirm_dialog)