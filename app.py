import streamlit as st
import pandas as pd
import re
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==============================================================================
# BAGIAN 1: LOGIKA CBR (BACKEND)
# ==============================================================================

def clean_string(text):
    if pd.isna(text): return 0
    if isinstance(text, (int, float)): return int(text)
    match = re.search(r'(\d+)', str(text))
    return int(match.group(1)) if match else 0

def convert_gpu_type(gpu_string):
    if pd.isna(gpu_string): return 0
    gpu_str = str(gpu_string).lower()
    if any(k in gpu_str for k in ["nvidia", "amd", "rtx", "gtx", "radeon"]): return 2
    elif any(k in gpu_str for k in ["intel", "iris", "uhd"]): return 1
    return 0 

# --- Fungsi Binning ---
def convert_ram(v): return 1 if v <= 8 else 2 if v <= 16 else 3
def convert_storage(v): return 1 if v <= 512 else 2 if v <= 1024 else 3
def convert_price(v): return 1 if v <= 40000 else 2 if v <= 60000 else 3
def convert_rating(v): return 1 if v <= 65 else 2 if v <= 75 else 3
def convert_display(v): return 1 if v <= 14 else 2 if v <= 15.6 else 3

def specs_to_vector(specs):
    return [
        convert_price(specs["price"]), convert_ram(specs["Ram"]),
        convert_storage(specs["ROM"]), convert_gpu_type(specs["GPU"]),
        convert_rating(specs["spec_rating"]), convert_display(specs["display_size"])
    ]

def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a * a for a in v1))
    mag2 = math.sqrt(sum(b * b for b in v2))
    return dot / (mag1 * mag2) if mag1 * mag2 != 0 else 0

# --- Load Data ---
@st.cache_data 
def load_case_base(file_path):
    try:
        df = pd.read_pickle(file_path)
    except:
        try:
            df = pd.read_csv(file_path.replace('.pkl', '.csv'))
        except:
            return []

    case_base = []
    for _, row in df.iterrows():
        try:
            case = {
                "criteria": {
                    "price": float(row['price']),
                    "spec_rating": float(row['spec_rating']),
                    "Ram": clean_string(row['Ram']), 
                    "ROM": clean_string(row['ROM']), 
                    "GPU": str(row['GPU']),
                    "display_size": float(row['display_size'])
                },
                "recommendation": row['name'] 
            }
            case_base.append(case)
        except: continue
    return case_base

def find_similar_cases(case_base, new_criteria_vector):
    all_similarities = []
    for i, case in enumerate(case_base):
        case_vector = specs_to_vector(case["criteria"])
        similarity = cosine_similarity(new_criteria_vector, case_vector)
        all_similarities.append({
            "case_id": i,
            "laptop_name": case["recommendation"],
            "similarity": similarity,
            "specs": case["criteria"] 
        })
    return sorted(all_similarities, key=lambda x: x['similarity'], reverse=True)

def revise_by_elimination(all_scores):
    """Logika Eliminasi Redundansi (Harga)"""
    if not all_scores: return "Tidak ada data.", None, False
    
    best_similarity = all_scores[0]['similarity']
    candidates = [case for case in all_scores if abs(case['similarity'] - best_similarity) < 0.0001]
    
    is_revised = False
    if len(candidates) > 1:
        is_revised = True
        candidates.sort(key=lambda x: x['specs']['price'])
        
    return candidates[0]["laptop_name"], candidates[0], is_revised

# ==============================================================================
# BAGIAN 2: TAMPILAN WEB (STREAMLIT)
# ==============================================================================

def main():
    st.set_page_config(page_title="SPK Laptop", layout="wide")
    st.title("💻 Sistem Rekomendasi Laptop")
    
    # --- Sidebar ---
    st.sidebar.header("⚙️ Filter Kebutuhan")
    with st.sidebar.form("input_form"):
        in_budget = st.number_input("Budget Maksimal", 0, 100000, 50000)
        in_ram = st.selectbox("RAM (GB)", [4, 8, 16, 32], index=2)
        in_rom = st.selectbox("Storage (GB)", [256, 512, 1024], index=1)
        in_gpu = st.radio("Tipe GPU", ["Office (Integrated)", "Gaming (Dedicated)"])
        in_rating = st.slider("Rating Min", 0, 100, 70)
        in_display = st.selectbox("Layar (Inch)", [13.3, 14.0, 15.6, 17.3], index=2)
        submitted = st.form_submit_button("🔍 Cari Rekomendasi")

    if submitted:
        case_base = load_case_base("data.pkl")
        if not case_base:
            st.error("Gagal memuat database! Pastikan file 'data.pkl' atau 'data.csv' ada.")
            return

        user_gpu = "Nvidia" if "Gaming" in in_gpu else "Intel"
        user_specs = {
            "price": in_budget, "Ram": in_ram, "ROM": in_rom,
            "GPU": user_gpu, "spec_rating": in_rating, "display_size": in_display
        }

        vec_user = specs_to_vector(user_specs)
        all_scores = find_similar_cases(case_base, vec_user)

        if not all_scores:
            st.warning("Database kosong.")
            return

        # 2. SLICING: Ambil HANYA 10 Besar
        top_10_candidates = all_scores[:10]

        # 3. FILTER BUDGET
        valid_candidates = [x for x in top_10_candidates if x['specs']['price'] <= in_budget]

        # 4. LOGIKA OUTPUT
        if not valid_candidates:
            st.error(f"❌ **Tidak Ditemukan!**")
            st.write(f"Tidak ada laptop di Top 10 kemiripan yang harganya di bawah **Rp {in_budget:,}**.")
            st.write("Silakan naikkan budget atau turunkan spesifikasi.")
            
            st.divider()
            st.subheader("📋 Hasil Pencarian")
            # Tabel Kosong dengan Kolom Lengkap
            empty_df = pd.DataFrame(columns=['Ranking', 'Nama Laptop', 'Harga', 'Skor', 'RAM', 'Storage', 'Layar', 'GPU'])
            st.dataframe(empty_df, use_container_width=True)
            
        else:
            # REVISE: Urutkan berdasarkan HARGA TERMURAH
            valid_candidates.sort(key=lambda x: x['specs']['price'])
            best_pick = valid_candidates[0]
            
            # --- Tampilan Sukses ---
            c1, c2 = st.columns([1, 1])
            with c1:
                st.success("✅ Rekomendasi Terbaik")
                st.info(f"**{best_pick['laptop_name']}**")
                
                # Tampilkan Detail Lengkap di Kartu Utama
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Harga", f"{best_pick['specs']['price']:,}")
                    st.metric("RAM", f"{best_pick['specs']['Ram']} GB")
                    st.metric("Layar", f"{best_pick['specs']['display_size']} Inch")
                with col_b:
                    st.metric("Skor", f"{best_pick['similarity']:.4f}")
                    st.metric("Storage", f"{best_pick['specs']['ROM']} GB")
                    st.metric("GPU Type", best_pick['specs']['GPU'])

                st.caption(f"Dipilih dari {len(valid_candidates)} kandidat yang lolos filter budget.")

            with c2:
                st.subheader("Grafik Perbandingan")
                names = [x['laptop_name'][:15] for x in valid_candidates]
                scores = [x['similarity'] for x in valid_candidates]
                
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.barh(names[::-1], scores[::-1], color='skyblue')
                ax.set_xlabel("Skor Kemiripan")
                st.pyplot(fig)

            # --- Tabel Detail Lengkap ---
            st.divider()
            st.subheader("📋 Tabel Detail Rekomendasi")
            
            table_data = []
            for rank, item in enumerate(valid_candidates):
                row = item['specs'].copy()
                row['Nama Laptop'] = item['laptop_name']
                row['Skor Kemiripan'] = round(item['similarity'], 4)
                row['Ranking'] = rank + 1
                table_data.append(row)
            
            df_table = pd.DataFrame(table_data)
            
            # --- PERBAIKAN: Menambahkan kolom ROM dan Display ---
            # Pilih kolom yang mau ditampilkan
            target_cols = ['Ranking', 'Nama Laptop', 'price', 'Skor Kemiripan', 'Ram', 'ROM', 'display_size', 'GPU', 'spec_rating']
            
            # Filter hanya kolom yang ada (biar gak error)
            cols = [c for c in target_cols if c in df_table.columns]
            df_table = df_table[cols]
            
            # Rename agar bahasa Indonesia dan Rapi
            df_table.rename(columns={
                'price': 'Harga (IDR)',
                'Ram': 'RAM (GB)',
                'ROM': 'Storage (GB)',          # <--- Muncul
                'display_size': 'Layar (Inch)', # <--- Muncul
                'spec_rating': 'Rating'
            }, inplace=True)
            
            st.dataframe(
                df_table.style.format({
                    "Harga (IDR)": "{:,.0f}",
                    "Layar (Inch)": "{:.1f}"
                })
                .background_gradient(subset=['Skor Kemiripan'], cmap="Blues"),
                use_container_width=True,
                hide_index=True
            )

if __name__ == "__main__":
    main()