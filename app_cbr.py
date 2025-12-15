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
    # Urutkan berdasarkan skor tertinggi
    return sorted(all_similarities, key=lambda x: x['similarity'], reverse=True)

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

        # Proses Vektorisasi
        user_gpu = "Nvidia" if "Gaming" in in_gpu else "Intel"
        user_specs = {
            "price": in_budget, "Ram": in_ram, "ROM": in_rom,
            "GPU": user_gpu, "spec_rating": in_rating, "display_size": in_display
        }

        vec_user = specs_to_vector(user_specs)
        
        # 1. RETRIEVE: Dapatkan semua skor kemiripan
        all_scores = find_similar_cases(case_base, vec_user)

        if not all_scores:
            st.warning("Database kosong.")
            return

        # 2. SLICING: Ambil HANYA 10 Besar (Top 10 Similarity)
        #    Kita hanya peduli pada 10 laptop yang paling mirip speknya.
        top_10_candidates = all_scores[:10]

        # 3. FILTER BUDGET: Cek mana dari Top 10 yang masuk budget
        valid_candidates = [x for x in top_10_candidates if x['specs']['price'] <= in_budget]

        # 4. LOGIKA HASIL (REVISE & OUTPUT)
        
        if not valid_candidates:
            # KASUS A: Tidak ada yang masuk budget di dalam Top 10
            st.error(f"❌ **Tidak ditemukan!**")
            st.write(f"Dari 10 laptop yang spesifikasinya paling mirip dengan keinginan Anda, tidak ada yang harganya di bawah **Rp {in_budget:,}**.")
            
            # Tampilkan Tabel Kosong sesuai permintaan
            st.subheader("📋 Hasil Pencarian")
            st.dataframe(pd.DataFrame(columns=['Nama Laptop', 'Harga', 'Skor Kemiripan', 'RAM', 'GPU']))
            
        else:
            # KASUS B: Ada yang masuk budget
            # REVISE: Urutkan berdasarkan HARGA TERMURAH dari kandidat yang lolos
            valid_candidates.sort(key=lambda x: x['specs']['price'])
            
            best_pick = valid_candidates[0] # Ambil yang paling murah
            
            # --- Tampilan Sukses ---
            c1, c2 = st.columns([1, 1])
            with c1:
                st.success("✅ Rekomendasi Terbaik (Termurah di Top 10)")
                st.info(f"**{best_pick['laptop_name']}**")
                st.metric(label="Harga", value=f"{best_pick['specs']['price']:,}")
                st.metric(label="Skor Kemiripan", value=f"{best_pick['similarity']:.4f}")
                st.caption(f"Terpilih dari {len(valid_candidates)} kandidat yang masuk budget dalam Top 10 similarity.")

            with c2:
                st.subheader("Grafik Perbandingan (Valid Candidates)")
                # Grafik hanya menampilkan yang valid (masuk budget & top 10)
                names = [x['laptop_name'][:15] for x in valid_candidates]
                scores = [x['similarity'] for x in valid_candidates]
                
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.barh(names[::-1], scores[::-1], color='skyblue')
                ax.set_xlabel("Skor Kemiripan")
                st.pyplot(fig)

            # --- Tabel Detail ---
            st.divider()
            st.subheader("📋 Tabel Detail Rekomendasi (Sesuai Budget)")
            
            table_data = []
            for item in valid_candidates:
                row = item['specs'].copy()
                row['Nama Laptop'] = item['laptop_name']
                row['Skor Kemiripan'] = round(item['similarity'], 4)
                table_data.append(row)
            
            df_table = pd.DataFrame(table_data)
            
            # Format dan Tampilkan Tabel
            st.dataframe(
                df_table.style.format({"price": "{:,.0f}"})
                .background_gradient(subset=['Skor Kemiripan'], cmap="Blues"),
                use_container_width=True
            )

if __name__ == "__main__":
    main()