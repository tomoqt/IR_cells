Data Structure for SR-microFTIR Cell Measurements

Root folder: SRmicroFTIR_cellule/
│
├── HaCaT/          # Normal cells
│   ├── AMT/        # Molecular aminopterine treatment
│   │   ├── HaCaT_1-2AMT_complete-no-baseline.dat   (1:2 concentration)
│   │   ├── HaCaT_1-8AMT_complete-no-baseline.dat   (1:8 concentration)
│   │   └── HaCaT_1-100AMT_complete-no-baseline.dat (1:100 concentration)
│   │
│   └── NP/         # Gold nanoparticle conjugated aminopterine
│       ├── HaCaT_1-2NP_complete-no-baseline.dat
│       └── HaCaT_1-8NP_complete-no-baseline.dat
│
├── HeLa/           # Cancer cells
│   ├── AMT/        # Molecular aminopterine treatment
│   │   └── HeLa_1-2AMT_complete-no-baseline.dat
│   │
│   ├── NP/         # Gold nanoparticle conjugated aminopterine
│   │   └── HeLa_1-8NP_complete-no-baseline.dat
│   │
│   └── HeLa-controllo_complete-no-baseline.dat      # Control dataset

Data format:
- Column 1: Wavenumbers (4000-800 cm⁻¹)
- Columns 2+: Individual cell spectra 