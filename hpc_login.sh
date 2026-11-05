# https://hpc.itu.dk/scheduling/login_to_hpc/
# ssh <initials>@hpc.itu.dk

# Clone the repository
# git clone https://github.com/Brainiaxx/Brain-Harmony.git
# cd Brain-Harmony

# Copy files from local machine to HPC
# cd "<local_path>/Brain-Harmony"
# rsync -aP \
#   --exclude '.git/' \
#   --exclude '.cursor/' \
#   --exclude 'data/abide_i/raw/' \
#   --exclude 'data/abide_i/work/' \
#   --exclude 'experiments/' \
#   ./ <initials>@hpc.itu.dk:~/Brain-Harmony/
