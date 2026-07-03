import cv2
import numpy as np
import supervision as sv
from ultralytics import RTDETR
import time
import csv
import os
from collections import deque
from datetime import datetime

# --- CONFIGURATION ---
VIDEO_PATH = "overheadvid2.mp4"
LOG_FILE = "girnar_log.csv"
GRAPH_HISTORY_LENGTH = 100
DASHBOARD_WIDTH = 450  

# --- PROFESSIONAL THEME COLORS (BGR Format) ---
C_BG = (25, 25, 25)        
C_CARD = (45, 45, 45)      
C_TEXT_MAIN = (240, 240, 240) 
C_TEXT_SUB = (160, 160, 160)  
C_ACCENT = (255, 191, 0)   
C_POS = (100, 200, 100)    
C_NEG = (100, 100, 255)    
C_WARN = (0, 165, 255)     
C_LINE_A = (0, 255, 255)   # Yellow for Line A
C_LINE_B = (255, 0, 255)   # Magenta for Line B
# ---------------------

def initialize_log():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "Event", "ID", "Direction", "Net_On_Steps", "Total_Ascended", "Total_Descended"])

def log_event(event_type, tracker_id, direction, net_count, asc_total, desc_total):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, event_type, tracker_id, direction, net_count, asc_total, desc_total])

# --- UI HELPER FUNCTIONS ---
def draw_card(img, x, y, w, h, color=C_CARD, border_color=None):
    cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
    if border_color:
        cv2.rectangle(img, (x, y), (x + w, y + h), border_color, 1)

def draw_text(img, text, x, y, font_scale=0.6, color=C_TEXT_MAIN, thickness=1, align="left"):
    font = cv2.FONT_HERSHEY_SIMPLEX
    size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    
    draw_x = x
    if align == "center":
        draw_x = x - size[0] // 2
    elif align == "right":
        draw_x = x - size[0]

    cv2.putText(img, text, (draw_x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    return size[1] 

def draw_dashboard(frame, asc, desc, net, fps, history, event_log):
    h, w, _ = frame.shape
    
    dashboard = np.zeros((h, DASHBOARD_WIDTH, 3), dtype=np.uint8)
    dashboard[:] = C_BG

    draw_card(dashboard, 0, 0, DASHBOARD_WIDTH, 70, color=C_CARD)
    cv2.rectangle(dashboard, (0, 68), (DASHBOARD_WIDTH, 70), C_ACCENT, -1) 
    draw_text(dashboard, "GIRNAR ANALYTICS", 20, 35, font_scale=0.8, thickness=2, color=C_TEXT_MAIN)
    draw_text(dashboard, "RT-DETR INTEL XE EDITION", 20, 58, font_scale=0.42, color=C_TEXT_SUB)
    
    status_color = C_POS if fps > 4 else C_NEG
    draw_text(dashboard, "● LIVE", DASHBOARD_WIDTH - 20, 35, font_scale=0.5, color=status_color, align="right")
    draw_text(dashboard, f"{fps:.1f} FPS", DASHBOARD_WIDTH - 20, 58, font_scale=0.45, color=C_TEXT_SUB, align="right")

    grid_y = 90
    card_w = (DASHBOARD_WIDTH - 60) // 2
    card_h = 100

    draw_card(dashboard, 20, grid_y, card_w, card_h)
    draw_text(dashboard, "NET PEOPLE ON STEPS", 35, grid_y + 25, font_scale=0.4, color=C_TEXT_SUB)
    
    net_color = C_TEXT_MAIN
    if net > 50: net_color = C_WARN
    if net > 100: net_color = C_NEG
    
    draw_text(dashboard, str(net), 35, grid_y + 75, font_scale=1.8, thickness=3, color=net_color)

    draw_card(dashboard, 40 + card_w, grid_y, card_w, card_h)
    draw_text(dashboard, "TOTAL TRAFFIC", 55 + card_w, grid_y + 25, font_scale=0.4, color=C_TEXT_SUB)
    
    draw_text(dashboard, "UP", 55 + card_w, grid_y + 55, font_scale=0.4, color=C_POS)
    draw_text(dashboard, str(asc), 90 + card_w, grid_y + 58, font_scale=0.7, thickness=2, color=C_TEXT_MAIN)
    
    draw_text(dashboard, "DOWN", 55 + card_w, grid_y + 85, font_scale=0.4, color=C_NEG)
    draw_text(dashboard, str(desc), 110 + card_w, grid_y + 88, font_scale=0.7, thickness=2, color=C_TEXT_MAIN)

    graph_y = 220
    graph_h = 180
    draw_card(dashboard, 20, graph_y, DASHBOARD_WIDTH - 40, graph_h)
    draw_text(dashboard, "REAL-TIME OCCUPANCY TREND", 35, graph_y + 25, font_scale=0.4, color=C_TEXT_SUB)
    
    plot_x = 35
    plot_y = graph_y + 40
    plot_w = DASHBOARD_WIDTH - 70
    plot_h = graph_h - 50
    
    cv2.line(dashboard, (plot_x, plot_y + plot_h//2), (plot_x + plot_w, plot_y + plot_h//2), (60,60,60), 1)
    cv2.line(dashboard, (plot_x, plot_y + plot_h), (plot_x + plot_w, plot_y + plot_h), (60,60,60), 1)

    if len(history) > 1:
        max_val = max(history) if max(history) > 5 else 5
        min_val = min(history) if min(history) < 0 else 0
        val_range = max_val - min_val
        if val_range == 0: val_range = 1

        pts = []
        for i, val in enumerate(history):
            px = int(plot_x + (i / GRAPH_HISTORY_LENGTH) * plot_w)
            py = int((plot_y + plot_h) - ((val - min_val) / val_range) * plot_h)
            pts.append([px, py])
        
        pts_arr = np.array(pts, np.int32).reshape((-1, 1, 2))
        bottom_right = [pts[-1][0], plot_y + plot_h]
        bottom_left = [pts[0][0], plot_y + plot_h]
        fill_pts = np.concatenate((pts_arr, np.array([bottom_right, bottom_left]).reshape((-1, 1, 2))))
        
        cv2.fillPoly(dashboard, [fill_pts], (60, 40, 0)) 
        cv2.polylines(dashboard, [pts_arr], False, C_ACCENT, 2, cv2.LINE_AA)

    log_y = 430
    draw_text(dashboard, "RECENT EVENTS", 20, log_y, font_scale=0.45, color=C_TEXT_SUB)
    
    for i, entry in enumerate(event_log):
        row_y = log_y + 30 + (i * 25)
        indicator_color = C_POS if "UP" in entry or "Entered" in entry else C_NEG
        cv2.circle(dashboard, (30, row_y - 4), 3, indicator_color, -1)
        draw_text(dashboard, entry, 45, row_y, font_scale=0.45, color=C_TEXT_MAIN)

    combined = np.hstack((frame, dashboard))
    return combined

def main():
    initialize_log()
    
    # --- INTEL IRIS XE OPENVINO OPTIMIZATION FOR RT-DETR ---
    print("====================================")
    print("Preparing RT-DETR OpenVINO Backend...")
    openvino_model_dir = "rtdetr-l_openvino_model" 
    
    if not os.path.exists(openvino_model_dir):
        print("🛠️ First time setup: Compiling RT-DETR into Intel OpenVINO format.")
        print("⏳ This will take 2-4 minutes. Please wait...")
        temp_model = RTDETR("rtdetr-l.pt")
        temp_model.export(format="openvino")
        print("✅ Intel optimization complete!")
    
    print("🚀 Loading Intel Iris Xe Engine...")
    model = RTDETR(openvino_model_dir)
    print("====================================")

    # Restored: Using Video File
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file target '{VIDEO_PATH}'")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: Video file is empty or corrupted.")
        return
        
    # Standardize height to 720p for consistent line placement
    TARGET_HEIGHT = 720
    orig_h, orig_w, _ = frame.shape
    scale_factor = TARGET_HEIGHT / orig_h
    new_width = int(orig_w * scale_factor)

    # ===================================
    # DUAL LINE CONFIGURATION
    # ===================================
    LINE_A_Y = int(TARGET_HEIGHT * 0.45)
    LINE_B_Y = int(TARGET_HEIGHT * 0.70)

    tracker = sv.ByteTrack()
    track_states = {}

    ascending_count = 0
    descending_count = 0
    people_on_steps = 0

    prev_time = time.time()
    net_count_history = deque([0] * GRAPH_HISTORY_LENGTH, maxlen=GRAPH_HISTORY_LENGTH)
    event_log_display = deque(maxlen=10)

    print("Pipeline running. Dashboard Active. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret: break

        current_time = time.time()
        fps_calculated = 1 / (current_time - prev_time) if prev_time != 0 else 0
        prev_time = current_time

        # Resize video frame smoothly
        frame = cv2.resize(frame, (new_width, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)

        # --- NATIVE OPENVINO INFERENCE ---
        results = model(frame, conf=0.35, classes=[0], verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections=detections)

        active_ids_this_frame = set(detections.tracker_id) if detections.tracker_id is not None else set()

        if detections.tracker_id is not None:
            for i, tracker_id in enumerate(detections.tracker_id):
                bbox = detections.xyxy[i]
                
                # --- FEET TRACKING FIX ---
                # Tracking the feet prevents giant bounding boxes from breaking the line math
                anchor_y = int(bbox[3]) 

                if tracker_id not in track_states:
                    track_states[tracker_id] = {
                        "last_y": anchor_y,
                        "state": None
                    }
                    continue

                state = track_states[tracker_id]["state"]
                last_y = track_states[tracker_id]["last_y"]

                # ===================================
                # CROSS LINE A OR B FIRST TIME
                # ===================================
                if state is None:
                    if (last_y < LINE_A_Y and anchor_y >= LINE_A_Y):
                        track_states[tracker_id]["state"] = "A"
                    elif (last_y > LINE_B_Y and anchor_y <= LINE_B_Y):
                        track_states[tracker_id]["state"] = "B"

                # ===================================
                # ENTRY: A -> B
                # ===================================
                elif state == "A":
                    if anchor_y >= LINE_B_Y:
                        ascending_count += 1
                        people_on_steps += 1

                        event_log_display.appendleft(f"↑ ID {tracker_id} Entered")
                        log_event("Movement", tracker_id, "UP", people_on_steps, ascending_count, descending_count)
                        track_states[tracker_id]["state"] = "COUNTED"

                # ===================================
                # EXIT: B -> A
                # ===================================
                elif state == "B":
                    if anchor_y <= LINE_A_Y:
                        descending_count += 1
                        people_on_steps = max(0, people_on_steps - 1)

                        event_log_display.appendleft(f"↓ ID {tracker_id} Exited")
                        log_event("Movement", tracker_id, "DOWN", people_on_steps, ascending_count, descending_count)
                        track_states[tracker_id]["state"] = "COUNTED"

                track_states[tracker_id]["last_y"] = anchor_y

        for stored_id in list(track_states.keys()):
            if stored_id not in active_ids_this_frame:
                del track_states[stored_id]

        net_count_history.append(people_on_steps)

        # --- RENDERING (Video Side) ---
        annotated_frame = frame.copy()
        
        cv2.line(annotated_frame, (0, LINE_A_Y), (new_width, LINE_A_Y), C_LINE_A, 3)
        cv2.putText(annotated_frame, "LINE A", (20, LINE_A_Y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_LINE_A, 2)
        
        cv2.line(annotated_frame, (0, LINE_B_Y), (new_width, LINE_B_Y), C_LINE_B, 3)
        cv2.putText(annotated_frame, "LINE B", (20, LINE_B_Y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_LINE_B, 2)
        
        if len(detections) > 0:
            for i, (x1, y1, x2, y2) in enumerate(detections.xyxy.astype(int)):
                t_id = detections.tracker_id[i] if detections.tracker_id is not None else "N/A"
                
                # Check if this ID is in a COUNTED state
                is_counted = False
                if t_id in track_states and track_states[t_id]["state"] == "COUNTED":
                    is_counted = True
                
                color = C_POS if is_counted else C_TEXT_SUB 
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw the Red Dot on the feet
                anchor_x = int((x1 + x2) / 2)
                cv2.circle(annotated_frame, (anchor_x, y2), 5, (0, 0, 255), -1)

                label = f"ID {t_id}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

        # --- RENDERING (Dashboard Side) ---
        final_output = draw_dashboard(
            annotated_frame, 
            ascending_count, 
            descending_count, 
            people_on_steps, 
            fps_calculated, 
            list(net_count_history),
            list(event_log_display)
        )

        cv2.imshow("Girnar Crowd Analytics - Dual Line Pro", final_output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()