# @author: Abhijeet

# One step commit script to commit and push a new version to the git repository
 
# command : sh git_push.sh "Some comment on the latest version in quotes" 

# Pull the last updated version from the repository
git pull
# Add ANY new files to the current version that might be new
git add .

# Commit the current version to the local git repository
# $1 is the User comment, given at the command line, which gets updated 
git commit -am "$1"

# Repository is pushed into the master repository
git push origin master
