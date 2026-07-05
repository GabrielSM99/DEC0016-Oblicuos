import cv2
import os
import sys
import numpy as np
import torch
import streamlit as st
import pandas as pd
from ultralytics import YOLO

# Configuração da página Web do Streamlit
st.set_page_config(page_title="GEMARS - Processamento", layout="centered")

st.title("Monitoramento da Pesca Artesanal - Torres")
st.markdown("### Sistema de Processamento de imagens")
st.write("---")

DISTANCIA_MAX = 150   

@st.cache_resource
def carregar_modelo():
    """ Carrega o modelo YOLOv26 uma única vez na memória """
    return YOLO('bestyolo26.pt')

try:
    model_yolo = carregar_modelo()
    device_name = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
    st.sidebar.success(f"IA Carregada com Sucesso!\nHardware: {device_name}")
except Exception as e:
    st.sidebar.error(f"Erro ao carregar o modelo YOLO: {e}")

videos_arquivos = st.file_uploader(
    "Arraste e solte ou busque os vídeos de triagem (.mp4, .avi, .mov)", 
    type=["mp4", "avi", "mov"], 
    accept_multiple_files=True
)

if videos_arquivos:
    st.info(f"Total de {len(videos_arquivos)} vídeo(s) carregado(s) na fila de processamento.")
    
    if st.button("Iniciar Processamento de Inventário em Lote", type="primary"):
        
        status_global = st.empty()
        
        for idx_video, arquivo_atual in enumerate(videos_arquivos):
            nome_base = os.path.splitext(arquivo_atual.name)[0]
            
            pasta_output_video = os.path.join(os.path.abspath("."), nome_base)
            pasta_capturas = os.path.join(pasta_output_video, "capturas")
            
            if not os.path.exists(pasta_capturas):
                os.makedirs(pasta_capturas)
            
            status_global.markdown(f"### 🎬 Processando vídeo {idx_video + 1} de {len(videos_arquivos)}: **{arquivo_atual.name}**")
            
            caminho_temporario = f"temp_{arquivo_atual.name}"
            with open(caminho_temporario, "wb") as f:
                f.write(arquivo_atual.read())
            
            cap = cv2.VideoCapture(caminho_temporario)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps_video = int(cap.get(cv2.CAP_PROP_FPS))
            w_video = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h_video = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if fps_video == 0:
                fps_video = 5 
            
            video_output_path = os.path.join(pasta_output_video, f"output_{arquivo_atual.name}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_video = cv2.VideoWriter(video_output_path, fourcc, fps_video, (w_video, h_video))
            
            rastros = {}
            contagem_real = 0
            dados_planilha = []
            
            barra_progresso = st.progress(0)
            status_texto = st.empty()
            
            results = model_yolo.track(source=caminho_temporario, conf=0.85, stream=True, persist=True, tracker="bytetrack.yaml")
            
            frame_idx = 0
            for res in results:
                frame_idx += 1
                img_orig = res.orig_img.copy()     
                frame_desenho = img_orig.copy()     
                ids_atuais = []
                
                pct = int((frame_idx / total_frames) * 100)
                barra_progresso.progress(pct)
                status_texto.text(f"Quadro {frame_idx}/{total_frames} ({pct}%)")
                
                if res.boxes and res.boxes.id is not None:
                    boxes = res.boxes.xyxy.cpu().numpy()
                    ids_yolo = res.boxes.id.cpu().numpy().astype(int)
                    confs = res.boxes.conf.cpu().numpy()
                    
                    for box, id_yolo, conf_yolo in zip(boxes, ids_yolo, confs):
                        cx, cy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
                        ids_atuais.append(id_yolo)
                        
                        foi_reidentificado = False
                        for old_id, data in rastros.items():
                            dist = np.linalg.norm(np.array((cx, cy)) - np.array(data['centro']))
                            if dist < DISTANCIA_MAX: 
                                rastros[old_id].update({'centro': (cx, cy), 'frames_sumido': 0})
                                
                                if conf_yolo > rastros[old_id]['yolo_max_conf']:
                                    rastros[old_id].update({
                                        'yolo_max_conf': conf_yolo,
                                        'best_frame': img_orig.copy(), 
                                        'best_box': list(map(int, box)),
                                        'frame_pico': frame_idx
                                    })
                                foi_reidentificado = True
                                break
                                
                        if not foi_reidentificado:
                            rastros[id_yolo] = {
                                'centro': (cx, cy), 'frames_sumido': 0, 'yolo_max_conf': conf_yolo,
                                
                                'best_frame': img_orig.copy(), 'best_box': list(map(int, box)), 'frame_pico': frame_idx
                            }
                            
                        x1, y1, x2, y2 = list(map(int, box))
                        cv2.rectangle(frame_desenho, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame_desenho, f"ID: {id_yolo}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                out_video.write(frame_desenho)
                
                ids_para_deletar = []
                for rid in list(rastros.keys()):
                    if rid not in ids_atuais:
                        rastros[rid]['frames_sumido'] += 1
                        if rastros[rid]['frames_sumido'] > 2:
                            peixe_dados = rastros[rid]
                            if peixe_dados['best_frame'] is not None:
                                contagem_real += 1
                                max_conf = peixe_dados['yolo_max_conf']
                                f_pico = peixe_dados['frame_pico']
                                
                                tempo_total_segundos = f_pico / fps_video
                                minutos = int(tempo_total_segundos // 60)
                                segundos = int(tempo_total_segundos % 60)
                                horario_formatado = f"{minutos:02d}:{segundos:02d}"
                                
                                foto_final_auditoria = peixe_dados['best_frame'].copy()
                                bx1, by1, bx2, by2 = peixe_dados['best_box']
                                
                                cv2.rectangle(foto_final_auditoria, (bx1, by1), (bx2, by2), (0, 255, 0), 3)
                                cv2.putText(foto_final_auditoria, f"INVENTARIO: #{contagem_real:03d} | ID YOLO: {rid} | CONF: {max_conf*100:.1f}%", 
                                            (bx1, by1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                
                                nome_imagem = f"peixe_n{contagem_real:03d}_id{rid}_completo.jpg"
                                caminho_print = os.path.join(pasta_capturas, nome_imagem)
                                cv2.imwrite(caminho_print, foto_final_auditoria)
                                
                                dados_planilha.append({
                                    "Nº Inventário": f"#{contagem_real:03d}",
                                    "ID Rastreio (YOLO)": rid,
                                    "Horário de Passagem (MM:SS)": horario_formatado,
                                    "Frame do Pico": f_pico,
                                    "Confiança Máxima da IA": f"{max_conf * 100:.2f}%",
                                    "Nome do Arquivo de Imagem": nome_imagem
                                })
                            ids_para_deletar.append(rid)
                            
                for d in ids_para_deletar:
                    del rastros[d]
                    
            cap.release()
            out_video.release()
            
            if os.path.exists(caminho_temporario):
                os.remove(caminho_temporario)
                
            if dados_planilha:
                df = pd.DataFrame(dados_planilha)
                excel_output_path = os.path.join(pasta_output_video, f"inventario_{nome_base}.xlsx")
                df.to_excel(excel_output_path, index=False)
                
            st.toast(f"Vídeo '{arquivo_atual.name}' concluído!")
            
        status_global.empty()
        st.success(f"Processamento em lote finalizado! {len(videos_arquivos)} vídeo(s) analisado(s).")
        st.balloons()
        st.write("### 📁 Estrutura de Pastas de Saída Gerada:")
        st.info("Verifique a pasta principal do projeto. Foram geradas pastas individuais contendo a Planilha, o Vídeo com bounding boxes e a pasta `/capturas` para cada arquivo submetido.")