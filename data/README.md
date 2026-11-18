# Data folder (not committed)

Copy the contents of the organisers' `Student folder.zip` here so that the layout is:

```
data/
  train/<trip_id>/<trip_id>.track|.nodes|.arcs|.route      (80 trips)
  student_test/<trip_id>/...                               (10 trips)
  evaluator_test/<trip_id>/...                             (10 trips, only on demo day)
```

A different location can be used by setting the environment variable `PATHSNAP_DATA`.
