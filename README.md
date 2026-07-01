# R2R – Data Repository

R2R (Rate-to-Rank) is a peer-assessment system that combines numeric grading with ranking-based aggregation.  
This repository contains anonymized data from the R2R system and code used for analysis. 

## Publications

### 1. Recency Effects in Pairwise Comparisons during Peer Assessment of Oral Presentations  
E. Yaakobi, I. Shalev, and L. Dery, “Recency effects in pairwise comparisons during peer assessment of oral presentations,” Assessment & Evaluation in Higher Education, 2026, pp. 1–8. [The paper](https://www.tandfonline.com/doi/full/10.1080/02602938.2026.2636700)


### 2. Cognitive-Aware Peer Assessment: Design Implications from a Classroom Deployment  
N. Bouskila and L. Dery, "Cognitive-Aware Peer Assessment: Design Implications from a Classroom Deployment," 2025 20th Conference on Computer Science and Intelligence Systems (FedCSIS), Kraków, Poland, 2025, pp. 641-646, doi: 10.15439/2025F8584.  
[The paper](https://ieeexplore.ieee.org/document/11214692)

### 3. Interactive and Iterative Peer Assessment  
L. Dery, “Interactive and Iterative Peer Assessment,” in Proceedings of the European Conference on Artificial Intelligence (ECAI 2024), FAIA, vol. 392, pp. 1519–1526, 2024. doi: 10.3233/FAIA240656.
[The paper](https://www.ariel.ac.il/wp/lihi-dery/wp-content/uploads/sites/84/2026/01/Dery_ECAI_2024-1.pdf)

### 4. Mitigating Generosity Bias in Peer Assessment: A Tool for Oral Class Presentations  
L. Dery and M. Lange, "Mitigating Generosity Bias in Peer Assessment: a Tool for Oral Class Presentations," 2024 IEEE International Conference on Advanced Learning Technologies (ICALT), Nicosia, North Cyprus, Cyprus, 2024, pp. 274-276, doi: 10.1109/ICALT61570.2024.00086.  
[The paper](https://ieeexplore.ieee.org/document/10645899)

---

## Repository Structure

- `data/case_study_*/` - anonymized R2R datasets organized by case study.
- `data/r2r_ranking_results/` - derived peer-rating and ranking outputs used to compare aggregation methods.
- `data/instructor_rankings_and_ratings/` - instructor final grades and group-level instructor rating averages.
- `code/ranking_computation/` - scripts for building peer rating/ranking datasets and computing R2R, Borda, Copeland, mean, and median rankings.
- `code/IOL/` and `code/recency bias/` - analysis code for the IOL study (paper #2) and the recency bias study (paper #1).

---

## Data Notes

All datasets are anonymized and contain no identifying information.

### R2R ranking computation

The ranking-computation code is kept in `code/ranking_computation/`:

- `build_peer_rating_ranking_dataset.py` combines each peer numeric rating with the matching rank position from the same reviewer's ranking file. It writes `data/r2r_ranking_results/peer_rating_ranking_evaluations.csv`.
- `compute_r2r_rankings.py` computes group-level rankings for each session using `R2R_copeland`, `R2R_borda`, `Copeland`, `Borda`, `Mean`, and `Median`. It writes `data/r2r_ranking_results/session_method_rankings_long.csv` and `data/r2r_ranking_results/session_method_rankings_wide.csv`.

Derived R2R ranking files are organized under `data/r2r_ranking_results/`:

- `peer_rating_ranking_evaluations.csv` - long table of reviewer-level peer ratings paired with reviewer-level ranks.
- `session_method_rankings_long.csv` - long table of group scores and ranks for each aggregation method.
- `session_method_rankings_wide.csv` - wide comparison table of method ranks by group/session.

### Instructor ratings

Instructor-grade outputs are kept in `data/instructor_rankings_and_ratings/`:

- `instructor_final_grades.csv` contains the fixed anonymized instructor final-grade rows: `session`, `username`, `group_number`, and `final_grade`. 
- `instructor_group_ratings.csv` contains one row per session/group with the average instructor final grade, number of students, minimum final grade, and maximum final grade.

---

## Contact

Lihi Dery  
lihid@ariel.ac.il
