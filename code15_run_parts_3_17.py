# run_parts_3_17.py
import subprocess, os, sys

base = r"C:/Users/hbala/Desktop/Chagas_Physionet2025"
input_dir  = os.path.join(base, "code15_input")
output_dir = os.path.join(base, "code15_output")
demographics = os.path.join(input_dir, "exams.csv")
labels       = os.path.join(input_dir, "code15_chagas_labels.csv")

for part in range(0, 1):          # 0 → 1 inclusive
    in_file  = os.path.join(input_dir,  f"exams_part{part}.hdf5")
    out_path = os.path.join(output_dir, f"exams_part{part}")
    os.makedirs(out_path, exist_ok=True)

    cmd = [
        sys.executable, "prepare_code15_data.py",
        "-i", in_file,
        "-d", demographics,
        "-l", labels,
        "-o", out_path,
        "-f", "mat"
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
print("All parts complete.")
