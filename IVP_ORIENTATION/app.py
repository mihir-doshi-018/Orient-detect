"""
Factory Conveyor Interlocking Block Orientation Detection System
Frontend Interface & Real-Time Telemetry Analytics Suite
"""

import os
import tempfile
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Strict Constraint: Read-only black-box imports from backend processing modules
from processing.segmentation import segment_blocks
from processing.orientation import calculate_orientation
from processing.rotation import calculate_correction
from processing.tracking import CentroidTracker


# ==============================================================================
# 1. GLOBAL CONFIGURATION & INDUSTRIAL DARK THEME SETUP
# ==============================================================================
st.set_page_config(
    page_title="Factory Conveyor Block Inspection",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean, authentic dark-mode industrial CSS injection with background image
st.markdown("""
<style>
    /* Industrial Background Image with Dark Overlay */
    .stApp {
        background-image: linear-gradient(rgba(14, 17, 23, 0.92), rgba(14, 17, 23, 0.92)), url("https://images.unsplash.com/photo-1581091226825-a6a2a5aee158?q=80&w=2070&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: #e5e7eb;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1f2937;
    }

    /* Crisp Industrial Metric Cards with Cyan Accent Border */
    div[data-testid="stMetric"] {
        background-color: #161f30;
        border: 1px solid #1f2937;
        border-left: 4px solid #0ea5e9;
        border-radius: 6px;
        padding: 12px 16px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.4);
        transition: all 0.2s ease-in-out;
    }

    div[data-testid="stMetric"]:hover {
        border-color: #374151;
        border-left-color: #38bdf8;
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.12);
    }

    div[data-testid="stMetricLabel"] {
        color: #9ca3af !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    div[data-testid="stMetricValue"] {
        color: #f9fafb !important;
        font-size: 1.65rem !important;
        font-weight: 700 !important;
        font-variant-numeric: tabular-nums;
    }

    /* Clean Bolder Tab Navigation */
    .stTabs [data-baseweb="tab-list"] {
        border-bottom: 1px solid #1f2937;
        gap: 6px;
        padding-bottom: 2px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 10px 22px;
        color: #9ca3af;
        font-weight: 600;
        font-size: 0.92rem;
        letter-spacing: -0.01em;
        border: 1px solid transparent;
        background-color: transparent;
        transition: all 0.2s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #e5e7eb;
        background-color: rgba(31, 41, 55, 0.5);
    }

    .stTabs [aria-selected="true"] {
        background-color: #161f30 !important;
        border-color: #1f2937 #1f2937 transparent #1f2937 !important;
        color: #38bdf8 !important;
        font-weight: 700 !important;
        border-top: 2px solid #0ea5e9 !important;
    }

    /* Expander Dark Styling */
    div[data-testid="stExpander"] {
        background-color: #161f30;
        border: 1px solid #1f2937;
        border-radius: 6px;
        margin-bottom: 12px;
    }

    /* Dataframe Container Border */
    div[data-testid="stDataFrame"] {
        border: 1px solid #1f2937;
        border-radius: 6px;
    }

    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background-color: #0ea5e9;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. SESSION STATE MANAGEMENT (AUTOMATED VANISH & HISTORY TRACKING)
# ==============================================================================
if "inspection_results" not in st.session_state:
    st.session_state.inspection_results = []
if "all_history_results" not in st.session_state:
    st.session_state.all_history_results = []
if "video_history_meta" not in st.session_state:
    st.session_state.video_history_meta = []
if "inspection_completed" not in st.session_state:
    st.session_state.inspection_completed = False
if "last_processed_frame" not in st.session_state:
    st.session_state.last_processed_frame = None
if "active_file_sig" not in st.session_state:
    st.session_state.active_file_sig = None
if "run_counter" not in st.session_state:
    st.session_state.run_counter = 0
if "current_video_id" not in st.session_state:
    st.session_state.current_video_id = None
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "auto_run_pending" not in st.session_state:
    st.session_state.auto_run_pending = False


# ==============================================================================
# 3. INTERACTIVE CONTROL SIDEBAR (`st.sidebar`)
# ==============================================================================
st.sidebar.markdown("""
<div style="padding-bottom: 8px; border-bottom: 1px solid #1f2937; margin-bottom: 14px;">
    <div style="font-size: 1.15rem; font-weight: 700; color: #f9fafb; letter-spacing: -0.02em;">
        🏭 CONVEYOR VISION HUD
    </div>
    <div style="font-size: 0.75rem; color: #9ca3af;">
        Interlocking Block Orientation Detection
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.subheader("1. Video Stream Input")
uploaded_file = st.sidebar.file_uploader(
    "Upload Factory Conveyor Video (.mp4)",
    type=['mp4'],
    key=f"video_uploader_{st.session_state.uploader_key}",
    help="Select the factory conveyor video stream"
)

# Automated Vanish: When a new or different video is uploaded, previous video vanishes immediately!
if uploaded_file is None:
    if st.session_state.active_file_sig is not None:
        st.session_state.active_file_sig = None
        st.session_state.inspection_results = []
        st.session_state.inspection_completed = False
        st.session_state.last_processed_frame = None
        st.session_state.current_video_id = None
        st.session_state.auto_run_pending = False
else:
    file_signature = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.active_file_sig != file_signature:
        st.session_state.active_file_sig = file_signature
        st.session_state.inspection_results = []
        st.session_state.inspection_completed = False
        st.session_state.last_processed_frame = None
        st.session_state.run_counter += 1
        st.session_state.current_video_id = f"{uploaded_file.name} (Run #{st.session_state.run_counter})"
        st.session_state.auto_run_pending = True

    # Dedicated Delete / Unload button right under the file uploader
    if st.sidebar.button("🗑️ Delete / Unload Active Video", width="stretch", help="Unload the current video and wipe its active frame"):
        curr_id = st.session_state.current_video_id
        if curr_id:
            st.session_state.all_history_results = [
                r for r in st.session_state.all_history_results if r.get("Video_ID") != curr_id
            ]
            st.session_state.video_history_meta = [
                m for m in st.session_state.video_history_meta if m.get("Video_ID") != curr_id
            ]
        st.session_state.uploader_key += 1
        st.session_state.active_file_sig = None
        st.session_state.inspection_results = []
        st.session_state.last_processed_frame = None
        st.session_state.inspection_completed = True
        st.session_state.current_video_id = None
        st.session_state.auto_run_pending = False
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("2. Inspection Parameters")

tolerance = st.sidebar.slider(
    "Alignment Tolerance (± degrees)",
    min_value=0,
    max_value=15,
    value=3,
    step=1,
    help="Allowable angular threshold before marking block as requiring rotation"
)

st.sidebar.markdown("---")

# Compact Sidebar: Wrapped Dynamic ROI Boundaries inside an expander with 2x2 grid
with st.sidebar.expander("📐 Conveyor ROI Boundary Control", expanded=False):
    col_roi_x1, col_roi_x2 = st.columns(2)
    with col_roi_x1:
        roi_x_min = st.slider("Left Boundary (X Min %)", 0, 40, 8, step=1)
    with col_roi_x2:
        roi_x_max = st.slider("Right Boundary (X Max %)", 60, 100, 92, step=1)

    col_roi_y1, col_roi_y2 = st.columns(2)
    with col_roi_y1:
        roi_y_min = st.slider("Top Boundary (Y Min %)", 10, 60, 45, step=1)
    with col_roi_y2:
        roi_y_max = st.slider("Bottom Boundary (Y Max %)", 60, 100, 90, step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("3. Execution Controls")

col_btn1, col_btn2 = st.sidebar.columns(2)
with col_btn1:
    start_btn = st.button("▶️ Run Inspection", type="primary", width="stretch")
with col_btn2:
    reset_btn = st.button("🔄 Reset View", width="stretch")

if reset_btn:
    st.session_state.inspection_results = []
    st.session_state.last_processed_frame = None
    st.session_state.auto_run_pending = False
    st.rerun()

if start_btn:
    st.session_state.inspection_results = []
    st.session_state.last_processed_frame = None
    st.session_state.run_counter += 1
    if uploaded_file is not None:
        st.session_state.current_video_id = f"{uploaded_file.name} (Run #{st.session_state.run_counter})"

# Active Video Session Sidebar Status
if st.session_state.current_video_id:
    hist_count = len(st.session_state.video_history_meta)
    st.sidebar.markdown(f"""
    <div style="background-color: #161f30; border: 1px solid #1f2937; border-radius: 6px; padding: 10px 12px; margin-top: 14px;">
        <div style="font-size: 0.7rem; color: #9ca3af; text-transform: uppercase; font-weight: 600; letter-spacing: 0.05em;">Active Video Session</div>
        <div style="font-size: 0.82rem; color: #38bdf8; font-weight: 700; word-break: break-all; margin-top: 2px;">{st.session_state.current_video_id}</div>
        <div style="font-size: 0.72rem; color: #6b7280; margin-top: 4px;">Archived Video Runs: {hist_count}</div>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# 4. DASHBOARD HEADER & DYNAMIC SYSTEM STATUS
# ==============================================================================
should_run = (start_btn or st.session_state.auto_run_pending) and (uploaded_file is not None)

col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.markdown("""
    <div style="margin-bottom: 0.25rem;">
        <span style="font-size: 1.5rem; font-weight: 700; color: #f9fafb;">
            🏭 Interlocking Block Orientation Detection System
        </span>
    </div>
    <div style="font-size: 0.85rem; color: #9ca3af; margin-bottom: 1.25rem;">
        Computer Vision pipeline with shortest-path angular correction, centroid tracking, and real-time telemetry analytics.
    </div>
    """, unsafe_allow_html=True)

with col_head2:
    header_status = st.empty()
    if uploaded_file is None:
        status_label = "AWAITING VIDEO"
        status_color = "#9ca3af"
        status_bg = "rgba(107, 114, 128, 0.15)"
        status_border = "#374151"
    elif should_run:
        status_label = "PROCESSING..."
        status_color = "#f59e0b"
        status_bg = "rgba(245, 158, 11, 0.15)"
        status_border = "#d97706"
    elif st.session_state.inspection_completed:
        status_label = "INSPECTION COMPLETE"
        status_color = "#10b981"
        status_bg = "rgba(16, 185, 129, 0.15)"
        status_border = "#059669"
    else:
        status_label = "READY"
        status_color = "#38bdf8"
        status_bg = "rgba(14, 165, 233, 0.15)"
        status_border = "#0284c7"

    header_status.markdown(f"""
    <div style="text-align: right; padding-top: 6px;">
        <span style="font-size: 0.75rem; text-transform: uppercase; color: #6b7280; font-weight: 600; letter-spacing: 0.05em;">System Status:</span>
        <span style="background-color: {status_bg}; color: {status_color}; font-size: 0.8rem; font-weight: 700; padding: 5px 12px; border-radius: 4px; border: 1px solid {status_border}; margin-left: 6px; letter-spacing: 0.03em;">
            {status_label}
        </span>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# 5. MULTI-TAB DASHBOARD VIEW (`st.tabs`)
# ==============================================================================
tab_feed, tab_analytics = st.tabs([
    "🎥 Live Inspection Feed",
    "📊 Real-Time Analytics Dashboard"
])


# ------------------------------------------------------------------------------
# TAB 1: LIVE INSPECTION FEED
# ------------------------------------------------------------------------------
with tab_feed:
    # Real-Time KPI Summary Metric Cards Row
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    metric_frames = kpi1.empty()
    metric_active = kpi2.empty()
    metric_unique = kpi3.empty()
    metric_pass = kpi4.empty()
    metric_rework = kpi5.empty()

    has_data = len(st.session_state.inspection_results) > 0
    if has_data:
        saved_df = pd.DataFrame(st.session_state.inspection_results)
        aligned_total = len(saved_df[saved_df['Status'] == 'Aligned'])
        rework_total = len(saved_df[saved_df['Status'] == 'Needs Rotation'])
        metric_frames.metric("Inspected Frames", int(saved_df['Frame'].max()))
        metric_active.metric("Total Detections", len(saved_df))
        metric_unique.metric("Unique Tracked IDs", saved_df['ID'].nunique())
        metric_pass.metric("Aligned (Pass)", aligned_total)
        metric_rework.metric("Needs Rotation", rework_total)
    else:
        metric_frames.metric("Inspected Frames", 0)
        metric_active.metric("Active in ROI", 0)
        metric_unique.metric("Unique Tracked IDs", 0)
        metric_pass.metric("Aligned (Pass)", 0)
        metric_rework.metric("Needs Rotation", 0)

    # Refined Visual Divider between KPI metrics and Video Feed
    st.markdown("""
    <div style="display: flex; align-items: center; margin: 12px 0 16px 0;">
        <div style="flex-grow: 1; height: 1px; background: linear-gradient(90deg, rgba(14,165,233,0.3) 0%, rgba(31,41,55,0.8) 100%);"></div>
        <div style="padding: 0 12px; font-size: 0.72rem; color: #6b7280; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em;">
            Optical Inspection Viewport
        </div>
        <div style="flex-grow: 1; height: 1px; background: linear-gradient(90deg, rgba(31,41,55,0.8) 0%, rgba(14,165,233,0.3) 100%);"></div>
    </div>
    """, unsafe_allow_html=True)

    # Centered High-FPS Video Canvas Player
    stframe = st.empty()
    status_placeholder = st.empty()
    progress_placeholder = st.empty()

    if uploaded_file is None:
        if st.session_state.last_processed_frame is not None:
            stframe.image(st.session_state.last_processed_frame, channels="BGR", width="stretch")
        else:
            status_placeholder.markdown("""
            <div style="background-color: #131b2a; border: 1px solid #1f2937; border-left: 4px solid #38bdf8; border-radius: 6px; padding: 20px 24px; margin: 8px 0;">
                <div style="display: flex; align-items: flex-start; gap: 14px;">
                    <span style="font-size: 1.6rem; line-height: 1;">📥</span>
                    <div>
                        <div style="font-size: 1rem; font-weight: 700; color: #f9fafb; margin-bottom: 4px;">
                            Conveyor Stream Inactive
                        </div>
                        <div style="font-size: 0.85rem; color: #9ca3af; line-height: 1.5;">
                            Please upload a factory conveyor video (<code style="color: #38bdf8; background-color: #1f2937; padding: 2px 6px; border-radius: 4px;">.mp4</code>) in the sidebar control panel, then click <strong style="color: #38bdf8;">▶️ Run Inspection</strong> to initiate computer vision detection, angular orientation analysis, and real-time defect telemetry.
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        if should_run:
            st.session_state.auto_run_pending = False

            stframe.empty()
            status_placeholder.markdown("""
            <div style="background-color: #161f30; border: 1px solid #1f2937; border-radius: 6px; padding: 10px 14px; margin-bottom: 8px; color: #38bdf8; font-size: 0.85rem; font-weight: 600;">
                🔄 Initializing Vision Pipeline... Previous video stream cleared.
            </div>
            """, unsafe_allow_html=True)

            active_video_id = st.session_state.current_video_id or f"{uploaded_file.name} (Run #{st.session_state.run_counter})"
            st.session_state.current_video_id = active_video_id

            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            tfile.close()

            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Increased max_distance to 90 for stable ID prediction and reduced fragmentation
            tracker = CentroidTracker(max_distance=90)
            raw_results = []

            frame_count = 0
            aligned_count = 0
            rework_count = 0

            try:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1
                    if frame_count == 1:
                        status_placeholder.empty()

                    height, width, _ = frame.shape

                    # Dynamic Conveyor ROI polygon box from sidebar sliders
                    roi_pts = np.array([
                        [int(width * (roi_x_min / 100.0)), int(height * (roi_y_min / 100.0))],
                        [int(width * (roi_x_max / 100.0)), int(height * (roi_y_min / 100.0))],
                        [int(width * (roi_x_max / 100.0)), int(height * (roi_y_max / 100.0))],
                        [int(width * (roi_x_min / 100.0)), int(height * (roi_y_max / 100.0))]
                    ], dtype=np.int32)

                    # 1. Segment blocks inside ROI
                    contours, mask = segment_blocks(frame, roi_pts)
                    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    current_centroids = []
                    contour_data = []

                    for cnt in contours:
                        angle, centroid = calculate_orientation(cnt, frame_gray)
                        current_centroids.append(centroid)
                        contour_data.append((cnt, angle, centroid))

                    # 2. Tracking
                    tracked_objects = tracker.update(current_centroids)

                    # Adaptive display scaling
                    scale = max(1.0, width / 1280.0)
                    thickness = max(2, int(scale * 2))
                    font_scale = max(0.5, scale * 0.55)

                    active_blocks = 0

                    # 3. Draw & Calculate Rotation via Shortest Path
                    for obj_id, (cx, cy) in tracked_objects.items():
                        for cnt, angle, (px, py) in contour_data:
                            if (cx, cy) == (px, py):
                                active_blocks += 1

                                # Map Current Angle to 0 - 90 range (local quadrant angle)
                                cur_0_90 = round(angle % 90, 2)

                                # Calculate 0-45 shortest path correction and direction using rotation module
                                correction, direction = calculate_correction(angle)
                                is_aligned = correction <= tolerance

                                if is_aligned:
                                    aligned_count += 1
                                else:
                                    rework_count += 1

                                color = (0, 255, 0) if is_aligned else (0, 0, 255)
                                cv2.drawContours(frame, [cnt], -1, color, thickness)

                                rot_text = "Aligned" if is_aligned else f"Rot: {correction} deg {direction[:2]}"

                                length = int(40 * scale)
                                end_x = int(cx + length * np.cos(np.radians(angle)))
                                end_y = int(cy + length * np.sin(np.radians(angle)))
                                cv2.arrowedLine(frame, (cx, cy), (end_x, end_y), (255, 0, 0), thickness, tipLength=0.3)

                                cv2.putText(frame, f"ID: {obj_id}", (cx - int(20*scale), cy - int(30*scale)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
                                cv2.putText(frame, f"Cur: {cur_0_90} deg", (cx - int(20*scale), cy - int(15*scale)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), thickness)
                                cv2.putText(frame, rot_text, (cx - int(20*scale), cy + int(20*scale)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

                                raw_results.append({
                                    "Video_ID": active_video_id,
                                    "Frame": frame_count,
                                    "ID": obj_id,
                                    "Current_Angle_0_90": cur_0_90,
                                    "Raw_Angle_Deg": angle,
                                    "Correction_0_90": correction,
                                    "Direction": direction,
                                    "Status": "Aligned" if is_aligned else "Needs Rotation"
                                })

                    cv2.polylines(frame, [roi_pts], isClosed=True, color=(0, 255, 255), thickness=max(2, int(2.5 * scale)))
                    stframe.image(frame, channels="BGR", width="stretch")

                    # Update live metric cards
                    if frame_count % 2 == 0 or frame_count == 1:
                        if total_frames > 0:
                            progress_placeholder.progress(min(1.0, frame_count / total_frames))
                        metric_frames.metric("Current Frame", f"{frame_count}/{total_frames}")
                        metric_active.metric("Active in ROI", active_blocks)
                        metric_unique.metric("Unique Tracked IDs", tracker.next_object_id)
                        metric_pass.metric("Aligned (Pass)", aligned_count)
                        metric_rework.metric("Needs Rotation", rework_count)

                # Post-processing filter to ensure accurate ID predictions (filter out transient flicker IDs appearing < 4 frames)
                df_raw = pd.DataFrame(raw_results)
                if not df_raw.empty:
                    id_counts = df_raw['ID'].value_counts()
                    valid_ids = id_counts[id_counts >= 4].index
                    filtered_df = df_raw[df_raw['ID'].isin(valid_ids)].copy()
                    results = filtered_df.to_dict('records')
                else:
                    results = []

                # Save state when inspection completes
                st.session_state.inspection_results = results
                st.session_state.inspection_completed = True
                st.session_state.last_processed_frame = frame
                
                # Append into global history results
                st.session_state.all_history_results.extend(results)

                # Record summary metadata for this Video_ID in history archive
                fpy_val = round(aligned_count / len(results) * 100.0, 1) if len(results) > 0 else 0.0
                unique_blocks_final = pd.DataFrame(results)['ID'].nunique() if len(results) > 0 else 0
                meta_entry = {
                    "Video_ID": active_video_id,
                    "Timestamp": datetime.now().strftime("%H:%M:%S"),
                    "Inspected_Frames": frame_count,
                    "Tracked_Blocks": unique_blocks_final,
                    "Aligned_Pass": aligned_count,
                    "Needs_Rotation": rework_count,
                    "First_Pass_Yield": f"{fpy_val}%"
                }
                st.session_state.video_history_meta = [
                    m for m in st.session_state.video_history_meta if m.get("Video_ID") != active_video_id
                ]
                st.session_state.video_history_meta.append(meta_entry)

                progress_placeholder.empty()
                status_placeholder.success(f"✅ Processing Complete for {active_video_id}! Inspected {frame_count} frames. Recorded to History Archive.")

                # Update top right status badge to complete
                header_status.markdown("""
                <div style="text-align: right; padding-top: 6px;">
                    <span style="font-size: 0.75rem; text-transform: uppercase; color: #6b7280; font-weight: 600; letter-spacing: 0.05em;">System Status:</span>
                    <span style="background-color: rgba(16, 185, 129, 0.15); color: #10b981; font-size: 0.8rem; font-weight: 700; padding: 5px 12px; border-radius: 4px; border: 1px solid #059669; margin-left: 6px; letter-spacing: 0.03em;">
                        INSPECTION COMPLETE
                    </span>
                </div>
                """, unsafe_allow_html=True)

            finally:
                cap.release()
                try:
                    os.unlink(tfile.name)
                except Exception:
                    pass
        else:
            if st.session_state.last_processed_frame is not None:
                stframe.image(st.session_state.last_processed_frame, channels="BGR", width="stretch")
                status_placeholder.info("Inspection complete. Click '▶️ Run Inspection' to re-process.")
            else:
                status_placeholder.info("👈 Video loaded. Click '▶️ Run Inspection' in sidebar to start inspection.")


# ------------------------------------------------------------------------------
# TAB 2: REAL-TIME ANALYTICS DASHBOARD
# ------------------------------------------------------------------------------
with tab_analytics:
    has_history = (
        len(st.session_state.all_history_results) > 0
        or len(st.session_state.video_history_meta) > 0
        or len(st.session_state.inspection_results) > 0
    )

    if not has_history:
        st.info("ℹ️ No inspection telemetry recorded yet. Upload a conveyor video in the sidebar and run the inspection feed to populate analytics.")
    else:
        combined_records = list(st.session_state.all_history_results)
        if not combined_records and st.session_state.inspection_results:
            combined_records = list(st.session_state.inspection_results)

        full_df = pd.DataFrame(combined_records)
        available_video_ids = list(dict.fromkeys(full_df["Video_ID"].dropna().tolist())) if ("Video_ID" in full_df.columns and not full_df.empty) else []

        # ======================================================================
        # 1. VIDEO HISTORY ARCHIVE & DELETION MANAGEMENT
        # ======================================================================
        st.markdown("""
        <div style="font-size: 1.15rem; font-weight: 700; color: #f9fafb; margin-top: 4px; margin-bottom: 2px;">
            🗂️ Video Run History Archive & Session Management
        </div>
        <div style="font-size: 0.8rem; color: #9ca3af; margin-bottom: 12px;">
            Audit historical conveyor runs. View performance metrics per Video ID or delete individual video runs from history.
        </div>
        """, unsafe_allow_html=True)

        col_hist_table, col_hist_ops = st.columns([2.7, 1.3], gap="medium")

        with col_hist_table:
            if st.session_state.video_history_meta:
                meta_df = pd.DataFrame(st.session_state.video_history_meta)
                display_meta_df = meta_df.rename(columns={
                    "Video_ID": "Video ID",
                    "Timestamp": "Time",
                    "Inspected_Frames": "Frames",
                    "Tracked_Blocks": "Tracked Blocks",
                    "Aligned_Pass": "Aligned (Pass)",
                    "Needs_Rotation": "Needs Rotation",
                    "First_Pass_Yield": "FPY Yield"
                })
                st.dataframe(display_meta_df, width="stretch", hide_index=True)
            elif available_video_ids:
                fallback_summary = []
                for vid in available_video_ids:
                    vdf = full_df[full_df["Video_ID"] == vid]
                    v_pass = len(vdf[vdf['Status'] == 'Aligned'])
                    v_rework = len(vdf[vdf['Status'] == 'Needs Rotation'])
                    v_yield = f"{(v_pass / len(vdf) * 100.0):.1f}%" if len(vdf) > 0 else "0.0%"
                    fallback_summary.append({
                        "Video ID": vid,
                        "Frames": vdf['Frame'].max() if 'Frame' in vdf.columns else 0,
                        "Tracked Blocks": vdf['ID'].nunique() if 'ID' in vdf.columns else 0,
                        "Aligned (Pass)": v_pass,
                        "Needs Rotation": v_rework,
                        "FPY Yield": v_yield
                    })
                st.dataframe(pd.DataFrame(fallback_summary), width="stretch", hide_index=True)
            else:
                st.caption("No video summary records in history.")

        with col_hist_ops:
            st.markdown("<div style='font-size: 0.85rem; font-weight: 600; color: #e5e7eb; margin-bottom: 4px;'>Manage History</div>", unsafe_allow_html=True)
            if available_video_ids:
                del_target = st.selectbox(
                    "Select Video ID to delete:",
                    options=available_video_ids,
                    key="select_del_video_id"
                )
                if st.button("🗑️ Delete Selected Video ID", type="secondary", width="stretch"):
                    st.session_state.all_history_results = [
                        r for r in st.session_state.all_history_results if r.get("Video_ID") != del_target
                    ]
                    st.session_state.inspection_results = [
                        r for r in st.session_state.inspection_results if r.get("Video_ID") != del_target
                    ]
                    st.session_state.video_history_meta = [
                        m for m in st.session_state.video_history_meta if m.get("Video_ID") != del_target
                    ]
                    if st.session_state.current_video_id == del_target:
                        st.session_state.uploader_key += 1
                        st.session_state.active_file_sig = None
                        st.session_state.last_processed_frame = None
                        st.session_state.inspection_completed = True
                        st.session_state.current_video_id = None
                        st.session_state.auto_run_pending = False

                    if "select_del_video_id" in st.session_state:
                        del st.session_state["select_del_video_id"]

                    st.success(f"✅ Video ID '{del_target}' permanently deleted.")
                    st.rerun()

                if st.button("⚠️ Clear Entire History", width="stretch", help="Wipes all historical telemetry and archive"):
                    st.session_state.all_history_results = []
                    st.session_state.video_history_meta = []
                    st.session_state.inspection_results = []
                    st.session_state.last_processed_frame = None
                    st.session_state.inspection_completed = True
                    st.session_state.current_video_id = None
                    st.session_state.active_file_sig = None
                    st.session_state.auto_run_pending = False
                    st.session_state.uploader_key += 1

                    if "select_del_video_id" in st.session_state:
                        del st.session_state["select_del_video_id"]

                    st.success("✅ All history and video sessions cleared.")
                    st.rerun()
            else:
                st.caption("No Video IDs to delete.")

        st.markdown("---")

        # ======================================================================
        # 2. TELEMETRY SCOPE & DYNAMIC KPI METRICS
        # ======================================================================
        scope_options = ["All Videos (Aggregated)"] + available_video_ids
        selected_scope = st.selectbox(
            "📂 Telemetry Scope for Analytics:",
            options=scope_options,
            index=0 if len(available_video_ids) > 1 else (1 if len(available_video_ids) == 1 else 0),
            help="Select a specific Video Run to analyze, or view aggregated statistics across all runs"
        )

        if selected_scope == "All Videos (Aggregated)":
            df = full_df.copy()
        else:
            df = full_df[full_df["Video_ID"] == selected_scope].copy()

        # Dynamic KPI Summary Metrics
        total_detections = len(df)
        aligned_detections = len(df[df['Status'] == 'Aligned'])
        rework_detections = len(df[df['Status'] == 'Needs Rotation'])

        fpy_pct = (aligned_detections / total_detections * 100.0) if total_detections > 0 else 0.0

        rework_df = df[df['Status'] == 'Needs Rotation']
        avg_correction = rework_df['Correction_0_90'].mean() if len(rework_df) > 0 else 0.0
        max_correction = rework_df['Correction_0_90'].max() if len(rework_df) > 0 else 0.0
        unique_parts_count = df['ID'].nunique() if ('ID' in df.columns and len(df) > 0) else 0

        a1, a2, a3, a4 = st.columns(4)
        a1.metric(
            label="First-Pass Yield (FPY %)",
            value=f"{fpy_pct:.1f}%",
            delta=f"{fpy_pct - 85.0:+.1f}% vs 85% Target",
            help="Percentage of block instances that meet alignment tolerance without rework"
        )
        a2.metric(
            label="Average Correction Angle",
            value=f"{avg_correction:.1f}°",
            delta=f"Max: {max_correction:.1f}°",
            delta_color="off",
            help="Mean angular rotation needed for blocks requiring adjustment"
        )
        a3.metric(
            label="Unique Tracked Blocks",
            value=unique_parts_count,
            help="Total distinct physical blocks tracked across conveyor run"
        )
        a4.metric(
            label="Rework Required Instances",
            value=rework_detections,
            delta=f"{(100.0 - fpy_pct):.1f}% Defect Rate",
            delta_color="inverse",
            help="Total detection events where rotation adjustment is necessary"
        )

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # ======================================================================
        # 3. SINGLE UNIFIED MASTER INDUSTRIAL CHART
        # ======================================================================
        st.markdown("""
        <div style="background-color: #161f30; border: 1px solid #1f2937; border-top: 3px solid #38bdf8; border-radius: 8px; padding: 16px 20px 8px 20px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span style="font-size: 1.05rem; font-weight: 700; color: #f9fafb;">
                    📊 Operational Alignment & Angular Correction Profile
                </span>
                <span style="font-size: 0.72rem; background-color: #111827; color: #38bdf8; padding: 3px 8px; border-radius: 4px; border: 1px solid #1f2937; font-weight: 600;">
                    UNIFIED TELEMETRY MASTER
                </span>
            </div>
            <div style="font-size: 0.8rem; color: #9ca3af; margin-bottom: 12px;">
                Complete operational breakdown of conveyor blocks: Aligned (Pass), Rotate Clockwise (CW), and Rotate Counter-Clockwise (CCW).
            </div>
        """, unsafe_allow_html=True)

        count_aligned = len(df[df['Status'] == 'Aligned'])
        cw_df = df[(df['Status'] == 'Needs Rotation') & (df['Direction'] == 'Clockwise')]
        ccw_df = df[(df['Status'] == 'Needs Rotation') & (df['Direction'] == 'Counter-Clockwise')]
        other_df = df[(df['Status'] == 'Needs Rotation') & (~df['Direction'].isin(['Clockwise', 'Counter-Clockwise']))]

        count_cw = len(cw_df)
        count_ccw = len(ccw_df)
        count_other = len(other_df)

        categories = ["Aligned (Pass)", "Rotate Clockwise (CW)", "Rotate Counter-Clockwise (CCW)"]
        counts = [count_aligned, count_cw, count_ccw]
        colors = ["#10b981", "#0ea5e9", "#f59e0b"]

        avg_corrections = [
            0.0,
            cw_df['Correction_0_90'].mean() if len(cw_df) > 0 else 0.0,
            ccw_df['Correction_0_90'].mean() if len(ccw_df) > 0 else 0.0
        ]
        max_corrections = [
            0.0,
            cw_df['Correction_0_90'].max() if len(cw_df) > 0 else 0.0,
            ccw_df['Correction_0_90'].max() if len(ccw_df) > 0 else 0.0
        ]

        if count_other > 0:
            categories.append("Requires Rework (Other)")
            counts.append(count_other)
            colors.append("#f43f5e")
            avg_corrections.append(other_df['Correction_0_90'].mean() if len(other_df) > 0 else 0.0)
            max_corrections.append(other_df['Correction_0_90'].max() if len(other_df) > 0 else 0.0)

        total_chart_count = sum(counts)
        pcts = [(c / total_chart_count * 100.0) if total_chart_count > 0 else 0.0 for c in counts]

        bar_labels = [
            f"<b>{c:,}</b> blocks<br><span style='font-size:11px; color:#cbd5e1;'>({p:.1f}%)</span>" if c > 0 else ""
            for c, p in zip(counts, pcts)
        ]

        customdata = np.stack((pcts, avg_corrections, max_corrections), axis=-1)

        fig_single = go.Figure()

        fig_single.add_trace(go.Bar(
            x=categories,
            y=counts,
            text=bar_labels,
            textposition="auto",
            textfont=dict(color="#ffffff", size=13),
            marker=dict(
                color=colors,
                line=dict(color="rgba(255, 255, 255, 0.25)", width=1.5)
            ),
            customdata=customdata,
            hovertemplate=(
                "<b>%{x}</b><br><br>"
                "Total Count: <b>%{y} blocks</b><br>"
                "Operational Share: <b>%{customdata[0]:.1f}%</b><br>"
                "Average Correction: <b>%{customdata[1]:.1f}°</b><br>"
                "Max Correction: <b>%{customdata[2]:.1f}°</b>"
                "<extra></extra>"
            )
        ))

        fig_single.add_annotation(
            xref="paper", yref="paper",
            x=0.98, y=0.94,
            text=(
                f"<b>Quality Score (FPY):</b> <span style='color:#10b981; font-weight:bold;'>{fpy_pct:.1f}%</span><br>"
                f"<span style='color:#9ca3af;'>Total Sample: {total_chart_count:,} detections</span>"
            ),
            showarrow=False,
            align="right",
            font=dict(size=12, color="#e5e7eb"),
            bgcolor="#111827",
            bordercolor="#374151",
            borderwidth=1,
            borderpad=8,
            opacity=0.95
        )

        fig_single.update_layout(
            height=380,
            margin=dict(l=30, r=30, t=25, b=25),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(17, 24, 39, 0.6)",
            font=dict(color="#e5e7eb", family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto"),
            yaxis=dict(
                title=dict(text="Total Block Detections", font=dict(color="#9ca3af", size=12)),
                gridcolor="rgba(255, 255, 255, 0.08)",
                zerolinecolor="rgba(255, 255, 255, 0.15)",
                tickfont=dict(color="#9ca3af")
            ),
            xaxis=dict(
                gridcolor="rgba(0,0,0,0)",
                tickfont=dict(color="#f3f4f6", size=12, weight="bold")
            ),
            showlegend=False
        )

        st.plotly_chart(fig_single, width="stretch", config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

        # ======================================================================
        # 4. SEARCHABLE HISTORICAL TELEMETRY LOG
        # ======================================================================
        st.markdown("---")
        st.subheader("📋 Searchable Historical Telemetry Log & CSV Export")

        f1, f2, f3, f4 = st.columns(4)
        with f1:
            filter_video = st.selectbox(
                "Filter Video ID:",
                options=["All"] + available_video_ids,
                index=0
            )
        with f2:
            filter_status = st.selectbox("Filter Status:", ["All", "Aligned", "Needs Rotation"])
        with f3:
            filter_dir = st.selectbox("Filter Direction:", ["All"] + sorted([str(d) for d in df['Direction'].unique()]))
        with f4:
            filter_id = st.multiselect("Filter Tracked Block ID:", options=sorted(df['ID'].unique().tolist()) if 'ID' in df.columns else [])

        filtered_df = df.copy()
        if filter_video != "All":
            filtered_df = filtered_df[filtered_df['Video_ID'] == filter_video]
        if filter_status != "All":
            filtered_df = filtered_df[filtered_df['Status'] == filter_status]
        if filter_dir != "All":
            filtered_df = filtered_df[filtered_dir['Direction'] == filter_dir]
        if filter_id and 'ID' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['ID'].isin(filter_id)]

        cols_order = ["Video_ID", "Frame", "ID", "Current_Angle_0_90", "Raw_Angle_Deg", "Correction_0_90", "Direction", "Status"]
        existing_cols = [c for c in cols_order if c in filtered_df.columns]
        filtered_df = filtered_df[existing_cols]

        st.dataframe(
            filtered_df,
            width="stretch",
            hide_index=True
        )

        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Telemetry Log as CSV",
            data=csv,
            file_name='conveyor_block_orientation_telemetry.csv',
            mime='text/csv',
            width="stretch"
        )