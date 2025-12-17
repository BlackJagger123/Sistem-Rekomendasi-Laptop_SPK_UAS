import streamlit as st
import pandas as pd
import re
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def clean_price(text):
    if pd.isna(text): return 0.0
    text = str(text).replace('Rp', '').replace('.', '').replace(',', '').strip()
    match = re.search(r'(\d+)', text)
    return float(match.group(1)) if match else 0.0

def clean_spec(text):
    if pd.isna(text): return 0
    match = re.search(r'(\d+)', str(text))
    return int(match.group(1)) if match else 0

def convert_gpu_type(gpu_string):
    if pd.isna(gpu_string): return 0
    gpu_str = str(gpu_string).lower()
    if any(k in gpu_str for k in ["nvidia", "amd", "rtx", "gtx", "radeon"]): return 2
    elif any(k in gpu_str for k in ["intel", "iris", "uhd"]): return 1
    return 0 

def convert_ram(v): 
    if v <= 4: return 1
    elif v <= 8: return 2
    elif v <= 16: return 3
    else: return 4

def convert_storage(v): 
    return 1 if v <= 256 else 2 if v <= 512 else 3 if v <= 1024 else 4

def convert_price(v): 
    return 1 if v <= 40000 else 2 if v <= 60000 else 3

def convert_display(v):
    return int(round(v))

def specs_to_vector(specs):
    return [
        convert_price(specs["price"]), 
        convert_ram(specs["Ram"]),
        convert_storage(specs["ROM"]), 
        convert_gpu_type(specs["GPU"]),
        convert_display(specs["display_size"])
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
        df = pd.read_csv("data.csv")
    except:
        return []

    case_base = []
    for _, row in df.iterrows():
        try:
            p_val = clean_price(row.get('price', 0))
            
            case = {
                "criteria": {
                    "price": p_val,
                    "spec_rating": float(row.get('spec_rating', 0)),
                    "Ram": clean_spec(row.get('Ram', 0)), 
                    "ROM": clean_spec(row.get('ROM', 0)), 
                    "GPU": str(row.get('GPU', '')),
                    "display_size": float(row.get('display_size', 0))
                },
                "recommendation": row.get('name', 'Unknown')
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

def revise_by_elimination(valid_candidates):
    if not valid_candidates: return None, None, False
    valid_candidates.sort(key=lambda x: x['specs']['price'])
    winner = valid_candidates[0]
    is_revised = len(valid_candidates) > 1
    return winner["laptop_name"], winner, is_revised



def main():
    st.set_page_config(page_title="SPK Laptop", layout="wide")
    st.title("Sistem Rekomendasi Laptop")
    
    # --- Sidebar ---
    st.sidebar.header("Filter Kebutuhan")
    with st.sidebar.form("input_form"):
        in_budget = st.number_input("Budget Maksimal (Rp)", 0, 1000000, 10000, step=500000)
        in_ram = st.selectbox("RAM Minimal (GB)", [4, 8, 16, 32], index=1)
        in_rom = st.selectbox("Storage Minimal (GB)", [256, 512, 1024], index=1)
        in_gpu = st.radio("Tipe GPU", ["Office (Integrated)", "Gaming (Dedicated)"])
        in_display = st.selectbox("Layar (Inch)", [13.3, 14.0, 15.6, 16.0, 17.3], index=2)
        submitted = st.form_submit_button("Cari Rekomendasi")

    if submitted:
        case_base = load_case_base("data.csv")
        if not case_base:
            st.error("File data.csv tidak ditemukan.")
            return

        filtered_candidates = []
        target_gpu_type = 2 if "Gaming" in in_gpu else 1
        
        for case in case_base:
            specs = case['criteria']
            
            # Syarat Mutlak
            check_budget = specs['price'] <= in_budget
            check_ram = specs['Ram'] >= in_ram
            check_rom = specs['ROM'] >= in_rom
            
            current_gpu_type = convert_gpu_type(specs['GPU'])
            check_gpu = current_gpu_type == target_gpu_type
            
            # CEK LAYAR (Toleransi 0.5 inch agar 14.0 dan 14.1 dianggap sama)
            check_display = abs(specs['display_size'] - in_display) <= 0.5
            
            if check_budget and check_ram and check_rom and check_gpu and check_display:
                filtered_candidates.append(case)

        if not filtered_candidates:
            st.error(f"**Tidak Ditemukan!**")
            st.warning("Saran: Naikkan budget atau ubah ukuran layar.")
            
            st.divider()
            empty_df = pd.DataFrame(columns=['Nama', 'Harga', 'RAM', 'Storage', 'Layar', 'GPU'])
            st.dataframe(empty_df, use_container_width=True)
            return

        user_specs = {
            "price": in_budget, "Ram": in_ram, "ROM": in_rom,
            "GPU": "Nvidia" if target_gpu_type == 2 else "Intel", 
            "spec_rating": 0, 
            "display_size": in_display
        }
        vec_user = specs_to_vector(user_specs)
        
        final_scores = find_similar_cases(filtered_candidates, vec_user)
        
        # REVISE (Cari Termurah)
        top_10_filtered = final_scores[:10]
        final_name, final_data, revised = revise_by_elimination(top_10_filtered)
        
        # Tampilan
        c1, c2 = st.columns([1, 1])
        with c1:
            st.success("Rekomendasi Terbaik")
            st.info(f"**{final_name}**")
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Harga", f"{final_data['specs']['price']:,.0f}")
                st.metric("RAM", f"{final_data['specs']['Ram']} GB")
                st.metric("Layar", f"{final_data['specs']['display_size']} Inch")
            with col_b:
                st.metric("Skor", f"{final_data['similarity']:.4f}")
                st.metric("Storage", f"{final_data['specs']['ROM']} GB")

        with c2:
            st.subheader("Grafik Kandidat Terpilih")
            display_list = top_10_filtered[:10]
            names = [x['laptop_name'][:15] for x in display_list]
            scores = [x['similarity'] for x in display_list]
            
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.barh(names[::-1], scores[::-1], color='green')
            ax.set_xlabel("Skor Kemiripan")
            st.pyplot(fig)

        st.divider()
        st.subheader("Detail Rekomendasi")
        
        table_data = []
        for rank, item in enumerate(top_10_filtered[:20]):
            row = item['specs'].copy()
            row['Nama Laptop'] = item['laptop_name']
            row['Skor'] = round(item['similarity'], 4)
            row['Ranking'] = rank + 1
            table_data.append(row)
        
        df_table = pd.DataFrame(table_data)
        
        target = ['Ranking', 'Nama Laptop', 'price', 'Skor', 'Ram', 'ROM', 'display_size', 'GPU']
        cols = [c for c in target if c in df_table.columns]
        df_table = df_table[cols]
        df_table.rename(columns={'price':'Harga (IDR)', 'Ram':'RAM', 'ROM':'Storage', 'display_size':'Layar'}, inplace=True)
        
        st.dataframe(
            df_table.style.format({"Harga (IDR)": "{:,.0f}", "Layar": "{:.1f}"})
            .background_gradient(subset=['Skor'], cmap="Greens"),
            use_container_width=True,
            hide_index=True
        )

if __name__ == "__main__":
    main()