# Hasilkan 100 sampel bootstrap
n_samples = len(X_train)
n_bootstraps = 100
all_bootstrap_indices = []
all_oob_indices = []

np.random.seed(42) 
for i in range(n_bootstraps):
    # Hasilkan indeks sampel bootstrap
    bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
    
    # Temukan indeks OOB
    oob_indices = list(set(range(n_samples)) - set(bootstrap_indices))
    
    all_bootstrap_indices.append(bootstrap_indices)
    all_oob_indices.append(oob_indices)

# Detail cetak untuk sampel 1, 2, dan 100
samples_to_show = [0, 1, 99]

for i in samples_to_show:
    print(f"\nBootstrap Sample {i+1}:")
    print(f"Chosen indices: {sorted(all_bootstrap_indices[i])}")
    print(f"Number of unique chosen indices: {len(set(all_bootstrap_indices[i]))}")
    print(f"OOB indices: {sorted(all_oob_indices[i])}")
    print(f"Number of OOB samples: {len(all_oob_indices[i])}")
    print(f"Percentage of OOB: {len(all_oob_indices[i])/n_samples*100:.1f}%")
