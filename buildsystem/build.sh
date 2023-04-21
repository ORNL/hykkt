#!/bin/bash

export SRCDIR=${SRCDIR:-$PWD}

exit() {
  # Clear all trap handlers so this isn't echo'ed multiple times, potentially
  # throwing off the CI script watching for this output
  trap - `seq 1 31`

  # If called without an argument, assume not an error
  local ec=${1:-0}

  # Echo the snippet the CI script is looking for
  echo BUILD_STATUS:${ec}

  # Actually exit with that code, although it won't matter in most cases, as CI
  # is only looking for the string 'BUILD_STATUS:N'
  builtin exit ${ec}
}

# This will be the catch-all trap handler after arguments are parsed.
cleanup() {
  # Clear all trap handlers
  trap - `seq 1 31`

  # When 'trap' is invoked, each signal handler will be a curried version of
  # this function which has the first argument bound to the signal it's catching
  local sig=$1

  echo
  echo Exit code $2 caught in build script triggered by signal ${sig}.
  echo

  exit $2
}



echo "Usage: ./buildsystem/build.sh
----------------------------------------------------------------------
Clusters:

  By default, this script will attempt to determine the cluster it is being ran 
  on using the hostname command. If a known cluster is found, it's respective 
  script in the directory ./scripts/buildsystem will be sourced and the 
  variable MY_CLUSTER will be set. For example, on PNNL cluster Marianas, 
  hostname marianas.pnl.gov will be matched and 
  ./scripts/buildsystem/marianasVariables.sh will be sourced. If you would like 
  to add a cluster, create a script
  ./scripts/buildsystem/<my cluster>Variables.sh and specify the relevant
  environment variables. If the hostname is not correctly finding your cluster,
  you may specify MY_CLUSTER environment variable before running this script
  and the script will respect the environment variable. For example, on ORNL
  Ascent cluster, the hostname does not find the cluster, so we must specify
  MY_CLUSTER when running:

-----------------------------------------------------------------------
"
# Trap all signals and pass signal to the handler so we know what signal was
# sent in CI
for sig in `seq 1 31`; do
  trap "cleanup $sig \$? \$LINENO" $sig
done

set -xv

if [[ ! -v MY_CLUSTER ]]
then
  export MY_CLUSTER=`uname -n | sed -e 's/[0-9]//g' -e 's/\..*//'`
fi

export MY_CLUSTER_LOWER=$(echo ${MY_CLUSTER} | tr '[:upper:]' '[:lower:]')

# Correctly identify clusters based on hostname
case $MY_CLUSTER_LOWER in
  newell*)
    export MY_CLUSTER=newell
    ;;
  dl*|marianas|*fat*)
    export MY_CLUSTER=marianas
    ;;
  deception*)
    export MY_CLUSTER=deception
    ;;
  *)
    echo "Cluster $MY_CLUSTER not identified - you'll have to set relevant variables manually."
    ;;
esac

ulimit -s unlimited || echo 'Could not set stack size to unlimited.'
ulimit -l unlimited || echo 'Could not set max locked memory to unlimited.'

. /etc/profile.d/modules.sh
module purge

varfile="$SRCDIR/buildsystem/$(echo $MY_CLUSTER)-env.sh"


if [[ -f "$varfile" ]]; then
  source "$varfile"
  echo Sourced system-specific variables for $MY_CLUSTER
fi

module list

mkdir -p build

rm -rf build/*

cmake -B build -S . && 

cmake --build build

cd build
ctest -VV

exit $?
