import os
import cv2
import torch
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp

import config



# ==========================================================
# CONFIG
# ==========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 512
WINDOW_SECONDS = 1.0

# ==========================================================
# LOAD MODEL
# ==========================================================

def load_model(model_path):
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1
    )
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


# ==========================================================
# EXTRAER CSV DESDE INYECCION
# ==========================================================

def extract_csv(video_path, model_pipeta, model_ovocito, output_csv, safe_injection,retroceso_frame):

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, safe_injection,)

    ret, frame = cap.read()
    if not ret:
        print("No se pudo leer el frame inicial")
        return False

    h, w = frame.shape[:2]
    prev_gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5,5), 0)

    frame_index = safe_injection
    data = []

    MEDIUM_THRESHOLD = 20
    STRONG_THRESHOLD = 40

    while frame_index <= retroceso_frame - 20:

        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE)) / 255.0

        tensor = torch.from_numpy(rgb_resized)\
                    .permute(2,0,1)\
                    .unsqueeze(0)\
                    .float()\
                    .to(DEVICE)

        with torch.no_grad():
            pip = model_pipeta(tensor).sigmoid().cpu().numpy()[0,0]
            ovo = model_ovocito(tensor).sigmoid().cpu().numpy()[0,0]

        pip = cv2.resize(pip, (w,h))
        mask_pip = (pip > 0.3).astype(np.uint8)

        ovo = cv2.resize(ovo, (w,h))
        mask_ovo = (ovo > 0.5).astype(np.uint8)

        contours_ovo, _ = cv2.findContours(
            mask_ovo.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        ovo_cx, ovo_cy, ovo_radius = None, None, None

        if len(contours_ovo) > 0:
            c = max(contours_ovo, key=cv2.contourArea)
            if cv2.contourArea(c) > 500:
                (x,y), radius = cv2.minEnclosingCircle(c)
                ovo_cx = int(x)
                ovo_cy = int(y)
                ovo_radius = int(radius)

        ys, xs = np.where(mask_pip > 0)

        if len(xs) < 200:
            prev_gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5,5), 0)
            frame_index += 1
            continue

        pts = np.column_stack((xs, ys))

        mean = pts.mean(axis=0)
        pts_c = pts - mean

        _, _, vt = np.linalg.svd(pts_c, full_matrices=False)
        axis = vt[0] / np.linalg.norm(vt[0])

        # 🔥 Forzar eje hacia la derecha
        if axis[0] < 0:
            axis = -axis

        projections = pts @ axis

        # punta = extremo negativo (más interno)
        tip = pts[np.argmin(projections)].astype(int)

        debug_frame = frame.copy()

        # dibujar todos los puntos de máscara (opcional)
        # debug_frame[mask_pip == 1] = [0,255,0]

        # dibujar tip
        cv2.circle(debug_frame, (tip[0], tip[1]), 8, (0,0,255), -1)

        # dibujar eje
        length = 150
        x2 = int(tip[0] + axis[0] * length)
        y2 = int(tip[1] + axis[1] * length)

        cv2.line(debug_frame, (tip[0], tip[1]), (x2, y2), (255,0,0), 3)
        cv2.imwrite(f"debug/frame_{frame_index}.png", debug_frame)

       

        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5,5), 0)
        diff = cv2.absdiff(gray, prev_gray)
        prev_gray = gray.copy()

        diff[mask_pip == 0] = 0

        _, medium = cv2.threshold(diff, MEDIUM_THRESHOLD, 255, cv2.THRESH_BINARY)
        _, strong = cv2.threshold(diff, STRONG_THRESHOLD, 255, cv2.THRESH_BINARY)

        motion_mask = cv2.bitwise_or(medium, strong)

        contours, _ = cv2.findContours(
            motion_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for c in contours:

            if cv2.contourArea(c) < 5:
                continue

            M = cv2.moments(c)
            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            dx = cx - tip[0]
            dy = cy - tip[1]

            dist_axis = dx * axis[0] + dy * axis[1]

            data.append({
                "frame": frame_index,
                "motion_x": cx,
                "motion_y": cy,
                "tip_x": int(tip[0]),
                "tip_y": int(tip[1]),
                "axis_x": float(axis[0]),
                "axis_y": float(axis[1]),
                "dist_axis": float(dist_axis),
                "ovo_cx": ovo_cx,
                "ovo_cy": ovo_cy,
                "ovo_radius": ovo_radius,
            })

        frame_index += 1

    cap.release()

    if len(data) == 0:
        print("No se detectaron puntos")
        return False

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)

    print("CSV generado:", output_csv)
    return True


def analyze_csv(csv_path, video_path, start_frame, retroceso_frame, interno_frame, model_ovocito, output_dir, model):

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    df = pd.read_csv(csv_path)
    df_full = df.copy()
    df = df[df["frame"] >= start_frame].copy()

 
    # ======================================================
    # VELOCIDAD LATERAL DE LA PIPETA
    # ======================================================

    df_tip = df_full.sort_values("frame").copy()

    df_tip["tip_dx"] = df_tip["tip_x"].diff()
    df_tip["tip_dy"] = df_tip["tip_y"].diff()

    # vector perpendicular al eje
    df_tip["perp_x"] = -df_tip["axis_y"]
    df_tip["perp_y"] =  df_tip["axis_x"]

    # proyección lateral
    df_tip["lateral_speed"] = (
        df_tip["tip_dx"] * df_tip["perp_x"] +
        df_tip["tip_dy"] * df_tip["perp_y"]
    )

    df_tip["lateral_speed"] = df_tip["lateral_speed"].abs()

    df_speed = df_tip.groupby("frame")["lateral_speed"].mean().reset_index()

    if df.empty:
        print("CSV vacío")
        return

    # ======================================================
    # SEÑAL TEMPORAL
    # ======================================================
    df_frame = df.groupby("frame")["dist_axis"].median().reset_index()

    full_range = pd.DataFrame({
        "frame": range(df_frame["frame"].min(),
                       df_frame["frame"].max() + 1)
    })

    df_frame = full_range.merge(df_frame, on="frame", how="left")
    df_frame["dist_axis"] = df_frame["dist_axis"].interpolate()
    df_frame["smooth"] = df_frame["dist_axis"].rolling(5, center=True).mean()
    df_frame["derivative"] = df_frame["smooth"].diff()
    df_frame = df_frame.dropna().reset_index(drop=True)


    WINDOW_SIZE = int(fps * 1.0)

    best_score = -float('inf')
    best_start = None
    best_end = None

    window_debug = []

    for i in range(len(df_frame) - WINDOW_SIZE):

        window = df_frame.iloc[i:i+WINDOW_SIZE]

        dynamic_range = window["smooth"].max() - window["smooth"].min()

        center_frame = (window["frame"].iloc[0] + window["frame"].iloc[-1]) / 2
        distance_to_interno = abs(center_frame - interno_frame)

        # Score: prioriza rango dinámico alto y cercanía al frame interno (factor reducido)
        score = dynamic_range - distance_to_interno * 0.001  # factor reducido para menos prioridad

        slope_window = np.polyfit(
            window["frame"],
            window["smooth"],
            1
        )[0]

        window_debug.append({
            "start_frame": int(window["frame"].iloc[0]),
            "end_frame": int(window["frame"].iloc[-1]),
            "dynamic_range": dynamic_range,
            "distance_to_interno": distance_to_interno,
            "score": score,
            "slope": slope_window,
            "direction": "POSITIVE" if slope_window > 0 else "NEGATIVE"
        })

        if score > best_score:
            best_score = score
            best_start = int(window["frame"].iloc[0])
            best_end = int(window["frame"].iloc[-1])

    if best_start is None:
        print("No se detectó evento")
        return

    slope = np.polyfit(
        df_frame[(df_frame["frame"] >= best_start) &
                 (df_frame["frame"] <= best_end)]["frame"],
        df_frame[(df_frame["frame"] >= best_start) &
                 (df_frame["frame"] <= best_end)]["smooth"],
        1
    )[0]

    direction = "ASPIRACION" if slope > 0 else "LIBERACION"

    MIN_FRAMES_FROM_RETRO = 60  # 1 segundo a 60 fps

    if direction == "ASPIRACION" and retroceso_frame is not None:

        distance_to_retro = retroceso_frame - best_end

        if distance_to_retro < MIN_FRAMES_FROM_RETRO:
            print("[INFO] Aspiracion demasiado cerca del retroceso.")
            print("Distancia:", distance_to_retro, "frames")

            # Forzamos que no sea aspiracion válida
            direction = "LIBERACION"

    # ======================================================
    # VALIDACION: PIPETA NO DEBE MOVERSE MUCHO
    # ======================================================
    MAX_ALLOWED_LATERAL = 1.5  # ajustar

    if direction == "ASPIRACION":

        event_speed = df_speed[
            (df_speed["frame"] >= best_start) &
            (df_speed["frame"] <= best_end)
        ]["lateral_speed"].mean()

        if np.isnan(event_speed):
            event_speed = 0

        print("[DEBUG] Velocidad lateral media:", event_speed)

        if event_speed > MAX_ALLOWED_LATERAL:
            print("[INFO] Aspiracion descartada: movimiento lateral excesivo")
            direction = "LIBERACION"



    # ======================================================
    # EVENT DEBUG (siempre)
    # ======================================================
    cap.set(cv2.CAP_PROP_POS_FRAMES, best_start)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if config.SAVE_EVENT_DEBUG_VIDEO:
        out_event = cv2.VideoWriter(
            os.path.join(output_dir, "event_debug.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        for f_id in range(best_start, best_end+1):
            ret, frame = cap.read()
            if not ret:
                break

            cv2.putText(frame, direction,
                        (50,60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0,0,255),
                        3)

            out_event.write(frame)

        out_event.release()
    else:
        # consume frames without writing
        for f_id in range(best_start, best_end+1):
            ret, _ = cap.read()
            if not ret:
                break

    # ======================================================
    # DETECTAR ASPIRACION PREVIA (si liberacion)
    # ======================================================
    asp_start = None
    asp_end = None

    if direction == "ASPIRACION":
        asp_start = best_start
        asp_end = best_end

    elif direction == "LIBERACION":

        # ------------------------------------------
        # 1️⃣ Fin real de aspiración = máximo previo
        # ------------------------------------------
        
        before_release = df_frame[
            df_frame["frame"] < best_start
        ].copy()

        if len(before_release) < 10:
            asp_start = None
        else:

            # índice del máximo antes de la liberación
            peak_idx = before_release["smooth"].idxmax()

            asp_end = int(before_release.loc[peak_idx, "frame"])

            # ------------------------------------------
            # 2️⃣ Buscar inicio real de aspiración
            # ------------------------------------------

            # derivada discreta
            before_release["deriv"] = before_release["smooth"].diff()

            # solo hasta el pico
            up_to_peak = before_release[
                before_release["frame"] <= asp_end
            ].copy()

            # buscamos hacia atrás donde deja de ser creciente
            start_frame = None

            for i in range(len(up_to_peak)-2, 0, -1):

                # si derivada <= 0 durante varios frames → inicio
                if up_to_peak["deriv"].iloc[i] <= 0:

                    start_frame = up_to_peak["frame"].iloc[i+1]
                    break

            if start_frame is not None:
                asp_start = int(start_frame)
            else:
                asp_start = int(up_to_peak["frame"].iloc[0])

    artifact_frame = None
    artifact_second = None

    # ======================================================
    # ANALISIS ASPIRACION COMPLETO
    # ======================================================
    if asp_start is not None:
    
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, asp_start)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # create writers only if the corresponding flag is enabled
        out_soft = None
        out_heat = None
        out_seg = None

        if config.SAVE_ASPIRATION_SOFT_DEBUG:
            out_soft = cv2.VideoWriter(
                os.path.join(output_dir, "aspiration_soft_debug.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height)
            )

        if config.SAVE_ASPIRATION_HEATMAP_DEBUG:
            out_heat = cv2.VideoWriter(
                os.path.join(output_dir, "aspiration_heatmap_debug.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height)
            )

        if config.SAVE_ASPIRATION_SEGMENTED_DEBUG:
            out_seg = cv2.VideoWriter(
                os.path.join(output_dir, "aspiration_segmented_debug.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height)
            )

        ret, prev_frame = cap.read()
        if not ret:
            return

        prev_gray = cv2.GaussianBlur(
            cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY), (5,5), 0
        )

        frame_id = asp_start
        artifact_energy = []
        artifact_frames = []
        inside_counter = 0
        entry_confirmed = False
        real_asp_start = None
        STABLE_FRAMES = int(0.3 * fps)
        membrane_broken = False
        real_asp_start = None
        prev_ratio_inside = 0.0

        RATIO_JUMP_THRESHOLD = 0.08   # salto brusco típico de rotura
        MIN_RATIO_INSIDE = 0.05       # evitar ruido muy pequeño

        global_energy_series = []
        global_energy_frames = []

        fixed_axis = None
        fixed_bins = None
        fixed_segment_limits = None
        region_locked = False

        while frame_id <= asp_end:

            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE)) / 255.0

            tensor = torch.from_numpy(rgb_resized)\
                        .permute(2,0,1)\
                        .unsqueeze(0)\
                        .float()\
                        .to(DEVICE)

            with torch.no_grad():
                pip = model(tensor).sigmoid().cpu().numpy()[0,0]
                ovo = model_ovocito(tensor).sigmoid().cpu().numpy()[0,0]

            pip = cv2.resize(pip, (width, height))
            mask_pip = (pip > 0.3).astype(np.uint8)

            ovo = cv2.resize(ovo, (width, height))
            mask_ovo = (ovo > 0.5).astype(np.uint8)

            gray = cv2.GaussianBlur(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5,5), 0
            )

            diff_global = cv2.absdiff(gray, prev_gray)

            global_energy = np.sum(diff_global.astype(np.int32))
            global_energy_series.append(global_energy)
            global_energy_frames.append(frame_id)

            # ======== HEATMAP GLOBAL =========

            heat = cv2.normalize(diff_global, None, 0, 255, cv2.NORM_MINMAX)
            heat = heat.astype(np.uint8)
            heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

            overlay = cv2.addWeighted(frame, 0.7, heat, 0.3, 0)

            if out_heat is not None:
                out_heat.write(overlay)

            # =================================

            prev_gray = gray.copy()

            # ahora sí aplicamos máscara para análisis pipeta
            diff = diff_global.copy()
            diff[mask_pip == 0] = 0

            frame_seg = frame.copy()

            ys, xs = np.where(mask_pip > 0)
            pts = np.column_stack((xs, ys))

            best_segment = None
            energy = 0

            if len(pts) > 200:

                mean = pts.mean(axis=0)
                pts_c = pts - mean
                _, _, vt = np.linalg.svd(pts_c, full_matrices=False)
                axis = vt[0] / np.linalg.norm(vt[0])

                if axis[0] < 0:
                    axis = -axis

                projections = pts @ axis
                bins = np.linspace(projections.min(), projections.max(), 5)

                best_diff = 1.0

                # -------- calcular punta real --------
                tip = pts[np.argmin(projections)]
                tip_x = int(tip[0])
                tip_y = int(tip[1])

                # -------- detectar entrada estable --------
                if 0 <= tip_y < height and 0 <= tip_x < width:
                    if mask_ovo[tip_y, tip_x] > 0:
                        inside_counter += 1
                    else:
                        inside_counter = 0

                if not entry_confirmed and inside_counter >= STABLE_FRAMES:
                    entry_confirmed = True
                    real_asp_start = frame_id

               
                # 🔒 BLOQUEAR REGION SOLO UNA VEZ
                if entry_confirmed and not region_locked:

                    fixed_axis = axis.copy()

                    projections_fixed = pts @ fixed_axis

                    # 👉 dividir en 6 segmentos
                    fixed_bins = np.linspace(
                        projections_fixed.min(),
                        projections_fixed.max(),
                        7
                    )

                    best_diff = 1.0
                    fixed_bin_index = None

                    # 👉 SOLO segmentos centrales (1,2,3,4)
                    for i in range(1, 5):

                        mask_segment = (
                            (projections_fixed >= fixed_bins[i]) &
                            (projections_fixed <  fixed_bins[i+1])
                        )

                        segment_pts = pts[mask_segment]

                        if len(segment_pts) == 0:
                            continue

                        inside = 0
                        for p in segment_pts:
                            if mask_ovo[int(p[1]), int(p[0])] > 0:
                                inside += 1

                        ratio = inside / len(segment_pts)
                        diff_ratio = abs(ratio - 0.5)

                        if diff_ratio < best_diff:
                            best_diff = diff_ratio
                            fixed_bin_index = i

                    region_locked = True
                    print("REGION BLOQUEADA EN FRAME:", frame_id)

                # -------- dividir en 4 segmentos --------
                best_segment = None

                if region_locked and fixed_bin_index is not None:

                    projections_now = pts @ fixed_axis

                    min_proj = projections_now.min()
                    max_proj = projections_now.max()

                    bins_now = np.linspace(
                        projections_now.min(),
                        projections_now.max(),
                        7
                    )

                    low = bins_now[fixed_bin_index]
                    high = bins_now[fixed_bin_index + 1]

                    mask_segment = (projections_now >= low) & \
                                (projections_now < high)

                    best_segment = pts[mask_segment]

                # -------- pintar segmento frontera --------
                if best_segment is not None:
                    for p in best_segment:
                        x, y = int(p[0]), int(p[1])
                        frame_seg[y, x] = [0,255,0]

                    # -------- energía solo si entrada confirmada --------
                    if entry_confirmed:
                        coords_y = best_segment[:,1].astype(int)
                        coords_x = best_segment[:,0].astype(int)

                        energy = np.sum(diff[coords_y, coords_x].astype(np.int32))

                # -------- dibujar punta --------
                cv2.circle(frame_seg, (tip_x, tip_y), 5, (0,0,255), -1)

            # -------- guardar energía --------
            if real_asp_start is not None and frame_id >= real_asp_start:
                artifact_energy.append(energy)
                artifact_frames.append(frame_id)
            else:
                artifact_energy.append(0)
                artifact_frames.append(frame_id)

            # -------- mostrar estado en debug --------
            if entry_confirmed:
                cv2.putText(frame_seg,
                            "DENTRO OVOCITO (ESTABLE)",
                            (50,100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0,255,0),
                            2)
            else:
                cv2.putText(frame_seg,
                            "ROMPIENDO MEMBRANA",
                            (50,100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0,0,255),
                            2)

            cv2.putText(frame_seg,
                        "ASPIRACION",
                        (50,60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0,255,255),
                        3)
            
            if real_asp_start is not None and frame_id >= real_asp_start:
                cv2.putText(frame_seg,
                            "POST-ROTURA (ANALIZANDO)",
                            (50,140),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0,255,0),
                            2)
            else:
                cv2.putText(frame_seg,
                            "PENETRACION / ROTURA",
                            (50,140),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0,0,255),
                            2)

            if out_seg is not None:
                out_seg.write(frame_seg)

            frame_id += 1
        # =========================================
        # DETECTAR FIN DE PENETRACION (RELAX)
        # =========================================

        global_energy_series = np.array(global_energy_series)

        if len(global_energy_series) > 0:

            # =========================================
            # DETECTAR FIN DE PENETRACION (ESTABLE)
            # =========================================

            global_energy_series = np.array(global_energy_series)

            real_asp_start = asp_start  # valor por defecto


            if len(global_energy_series) > 10:

                peak_idx = np.argmax(global_energy_series)
                peak_value = global_energy_series[peak_idx]

                relax_threshold = 0.30 * peak_value
                RELAX_STABLE_FRAMES = int(0.5 * fps)

                counter = 0
                relax_idx = None

                for i in range(peak_idx + 1, len(global_energy_series)):

                    if global_energy_series[i] < relax_threshold:
                        counter += 1
                    else:
                        counter = 0

                    if counter >= RELAX_STABLE_FRAMES:
                        relax_idx = i
                        break

                if relax_idx is not None:
                    real_asp_start = global_energy_frames[relax_idx]

                if real_asp_start >= asp_end:
                    print("[WARNING] real_asp_start >= asp_end, corrigiendo...")
                    real_asp_start = asp_start


            # =========================================
            # ANALIZAR ARTEFACTO DESDE POST-ROTURA
            # =========================================

            artifact_energy = []
            artifact_frames = []


            cap2 = cv2.VideoCapture(video_path)
            cap2.set(cv2.CAP_PROP_POS_FRAMES, real_asp_start)

            prev_gray = None
            frame_id = real_asp_start

            print("real_asp_start:", real_asp_start)
            print("asp_end:", asp_end)
            print("frames a analizar:", asp_end - real_asp_start)

            while frame_id <= asp_end:

                ret, frame = cap2.read()
                if not ret:
                    break

                gray = cv2.GaussianBlur(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5,5), 0
                )

                if prev_gray is None:
                    prev_gray = gray
                    frame_id += 1
                    continue

                diff = cv2.absdiff(gray, prev_gray)
                prev_gray = gray

                energy = np.sum(diff.astype(np.int32))

                artifact_energy.append(energy)
                artifact_frames.append(frame_id)

                frame_id += 1

            cap2.release()


            artifact_frame = None
            artifact_second = None

            if len(artifact_energy) > 0:

                artifact_energy = np.array(artifact_energy)
                idx = np.argmax(artifact_energy)

                artifact_frame = artifact_frames[idx]
                artifact_second = artifact_frame / fps

                print("Artifact detectado en frame:", artifact_frame)

                cap_img = cv2.VideoCapture(video_path)
                cap_img.set(cv2.CAP_PROP_POS_FRAMES, artifact_frame)

                ret, frame_art = cap_img.read()
                if ret:
                    cv2.imwrite(
                        os.path.join(output_dir, "frame_aspiracion.png"),
                        frame_art
                    )

                cap_img.release()


    # ======================================================
    # GUARDAR TXT FINAL
    # ======================================================
    lib_start_sec = best_start / fps
    lib_end_sec = best_end / fps



    # ==================================================
    # VIDEO DEBUG ASPIRACION SUAVE (si hubo liberacion)
    # ==================================================

    if direction == "LIBERACION" and asp_start is not None:

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, asp_start)

        # writers may have been recreated earlier, ensure they exist if flag enabled
        # (we already handled creation above)

        for f in range(asp_start, asp_end+1):

            ret, frame = cap.read()
            if not ret:
                break

            row = df_full[df_full["frame"] == f]

            if len(row) > 0:

                row = row.iloc[0]

                tip_x = int(row["tip_x"])
                tip_y = int(row["tip_y"])
                axis_x = float(row["axis_x"])
                axis_y = float(row["axis_y"])

                # 🔴 Punta
                cv2.circle(frame, (tip_x, tip_y), 6, (0,0,255), -1)

                # 🔵 Eje
                length = 150
                x2 = int(tip_x + axis_x * length)
                y2 = int(tip_y + axis_y * length)
                cv2.line(frame, (tip_x, tip_y), (x2, y2), (255,0,0), 3)

                # 🟢 Dirección positiva
                x3 = int(tip_x + axis_x * 50)
                y3 = int(tip_y + axis_y * 50)
                cv2.circle(frame, (x3, y3), 5, (0,255,0), -1)

            cv2.putText(frame,
                        "ASPIRACION SUAVE",
                        (50,60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0,255,255),
                        3)

            # only write if the writer was actually created
            if out_soft is not None:
                out_soft.write(frame)

        cap.release()
        if out_soft is not None:
            out_soft.release()

        if out_soft is not None:
            print("Video aspiracion suave guardado")
    
    return artifact_frame


def run_heatmap_analysis(video_path,
                         output_dir,
                         interno_frame,
                         retroceso_frame,
                         injection_frame,
                         model_pipeta,
                         model_ovocito):
    
    os.makedirs(output_dir, exist_ok=True)

    print("Inyección recibida desde main:", interno_frame)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    MIN_OFFSET_FRAMES = 30  # medio segundo si el vídeo es 60fps

    safe_injection = interno_frame 

    csv_path = os.path.join(output_dir, "data.csv")

    ok = extract_csv(
        video_path,
        model_pipeta,
        model_ovocito,
        csv_path,
        safe_injection,
        retroceso_frame
    )

    if ok:
        artifact_frame = analyze_csv(
            csv_path,
            video_path,
            interno_frame,  # Primero desde interno hacia adelante
            retroceso_frame,
            interno_frame,
            model_ovocito,
            output_dir,
            model_pipeta
        )

        # Si no se detectó, intentar desde el frame de inyección hacia adelante
        if artifact_frame is None:
            print("[INFO] No se detectó artefacto desde interno, intentando desde frame de inyección")
            artifact_frame = analyze_csv(
                csv_path,
                video_path,
                injection_frame,
                retroceso_frame,
                interno_frame,
                model_ovocito,
                output_dir,
                model_pipeta
            )

        # eliminar csv intermedio si no se desea conservar
        if not config.SAVE_CSV_DATA:
            try:
                os.remove(csv_path)
                print("CSV eliminado según configuración")
            except OSError:
                pass
    
    return artifact_frame