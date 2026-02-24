This is a customized pipeline developed for molecular dynamics trajectories using the Python FrustratometeR ([Github](https://github.com/HanaJaafari/Frustratometer))

This pipeline is used exclusively for mutational frustration (either single residue frustration, or pairwise mutational frustration). The goal is to retrieve and save Z scores (frustration index) for all amino acid alternatives. 

Current design is to using Bash scripts as outer wrappers, and use Python core to run Z score. 