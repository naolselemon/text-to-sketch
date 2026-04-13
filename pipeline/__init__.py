"""
Text-to-Sketch  Pipeline.

Stages
------
1  Lineart extraction (ControlNet LineartAnimeDetector)
2  Vectorization + RDP simplification
3  Stroke ordering (Directional / Greedy / TSP)
4  Sigma-Lognormal kinematics
5  stroke-5 formatting
    + Tok-Dict K-means codebook
"""
