#!/bin/bash
for port in 6274 6275 6277; do
  pids=$(lsof -ti :$port 2>/dev/null)
  if [ -n "$pids" ]; then
    echo "Killing processes on port $port: $pids"
    echo "$pids" | xargs kill -9
  else
    echo "No process found on port $port"
  fi
done
