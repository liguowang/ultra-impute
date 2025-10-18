## How to Build the File: `GRCh38_methyl_probes.info.tsv.gz`

This file is required by the `fill_gKNN()` function.  
It must include the **first five columns** (any additional columns will be ignored):

| Column | Description |
|---------|--------------|
| `chrom` | Chromosome name |
| `start` | Start position |
| `end`   | End position |
| `CpG ID` | CpG probe identifier |
| `List of CREs` | Candidate Regulatory Elements overlapping the CpG site |

---

### Sources of Candidate Regulatory Elements (CREs)

The **CREs** were compiled from several major genomic annotation databases:

#### 1. ENCODE cCRE (Candidate cis-Regulatory Elements)
- **Description:** ENCODE Registry of candidate cis-Regulatory Elements (cCREs) in the human genome  
- **Total Regions:** 926,535  
- **Data Last Updated:** 2020-05-20  
- **Data Link:** [ENCODE cCRE Combined](https://hgdownload.soe.ucsc.edu/gbdb/hg38/encode3/ccre/encodeCcreCombined.bb)

#### 2. ENCODE DNase I Clusters
- **Total Regions:** 2,129,159  
- **Data Last Updated:** 2022-11-07  

#### 3. FANTOM5 CAGE Peaks (±50 bp Extension)
- **Description:** Cap Analysis Gene Expression (CAGE) peaks mapping transcription start sites (TSS); ±50 bp indicates core promoter regions.  
- **Total Regions:** 210,250  
- **Source:** [FANTOM5 Consortium](https://fantom.gsc.riken.jp/5/)

#### 4. FANTOM5 Enhancers
- **Total Regions:** 63,285  

#### 5. GeneHancer (v5.10)
- **Description:** Integrated database of human regulatory elements and their target genes.  
- **Total Regions:** 393,207  
- **Source:** [GeneHancer](https://www.genecards.org/Guide/GeneHancer)

#### 6. RefSeq Functional Elements
- **Description:** NCBI RefSeq curated collection of known and predicted functional elements.  
- **Total Regions:** 5,756  
- **Source:** [NCBI RefSeq Functional Elements](https://www.ncbi.nlm.nih.gov/refseq/functionalelements/)

#### 7. VISTA Enhancers
- **Description:** Experimentally validated human and mouse enhancer elements.  
- **Total Regions:** 3,907  
- **Data Last Updated:** 2025-07-01  
- **Data Link:** [VISTA Enhancers (UCSC)](https://hgdownload.soe.ucsc.edu/gbdb/hg38/vistaEnhancers/vistaEnhancers.bb)  
- **Source:** [VISTA Enhancer Browser](https://enhancer.lbl.gov/)
