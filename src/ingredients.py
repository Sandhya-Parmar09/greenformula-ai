"""Case studies: a conventional ingredient and candidate bio-based swaps for the same job.

`biobased` and `source` are curated by hand from public supplier/INCI information, not predicted.
Mixtures are represented by one typical molecule (noted in `note`).
"""

CASES = [
    {
        "function": "Silky emollient (skin & hair)",
        "incumbent": {"name": "Cyclopentasiloxane (D5)", "smiles": "C[Si]1(C)O[Si](C)(C)O[Si](C)(C)O[Si](C)(C)O[Si](C)(C)O1",
                      "biobased": 0, "source": "Synthetic silicone",
                      "note": "Restricted in EU rinse-off products under REACH (persistence concerns)"},
        "alternatives": [
            {"name": "Squalane", "smiles": "CC(C)CCCC(C)CCCC(C)CCCCC(C)CCCC(C)CCCC(C)C", "biobased": 1, "source": "Sugarcane / olive / amaranth"},
            {"name": "Isoamyl laurate", "smiles": "CCCCCCCCCCCC(=O)OCCC(C)C", "biobased": 1, "source": "Plant fatty acid + fusel oil"},
            {"name": "Undecane", "smiles": "CCCCCCCCCCC", "biobased": 1, "source": "Plant-derived alkane (coconut/palm kernel)"},
            {"name": "Dicaprylyl carbonate", "smiles": "CCCCCCCCOC(=O)OCCCCCCCC", "biobased": 1, "source": "Plant-derived fatty alcohol"},
            {"name": "Coco-caprylate", "smiles": "CCCCCCCCCCCCOC(=O)CCCCCCC", "biobased": 1, "source": "Coconut", "note": "Modelled as lauryl caprylate"},
        ],
    },
    {
        "function": "Antioxidant (protects oils from going rancid)",
        "incumbent": {"name": "BHT", "smiles": "Cc1cc(C(C)(C)C)c(O)c(C(C)(C)C)c1", "biobased": 0, "source": "Petrochemical"},
        "alternatives": [
            {"name": "Ferulic acid", "smiles": "COc1cc(/C=C/C(=O)O)ccc1O", "biobased": 1, "source": "Upcycled rice-bran oil refining by-product (India)"},
            {"name": "Tocopherol (Vit. E)", "smiles": "Cc1c(C)c2OC(C)(CCCC(C)CCCC(C)CCCC(C)C)CCc2c(C)c1O", "biobased": 1, "source": "Vegetable oil distillate"},
        ],
    },
    {
        "function": "Musk fragrance note",
        "incumbent": {"name": "Galaxolide (HHCB)", "smiles": "CC1COCC2=CC3=C(C=C12)C(C(C3(C)C)C)(C)C", "biobased": 0, "source": "Petrochemical polycyclic musk",
                      "note": "Widely detected in rivers and sediments"},
        "alternatives": [
            {"name": "Ethylene brassylate", "smiles": "O=C1CCCCCCCCCCCC(=O)OCCO1", "biobased": 1, "source": "Brassylic acid from plant erucic acid"},
        ],
    },
]
