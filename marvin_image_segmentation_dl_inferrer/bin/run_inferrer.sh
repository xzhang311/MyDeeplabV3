#!/apollo/sbin/envroot bash
export PYTHONPATH=""
exec $ENVROOT/bin/run_inferrer.py "$@"
