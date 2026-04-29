import marimo

__generated_with = "0.22.5"
app = marimo.App(width="full")


app._unparsable_cell(
    r"""
    ¥import marimo as mo
    import pandas as pd
    import plotly.express as px
    from pathlib import Path
    """,
    name="_"
)


@app.cell
def _(Path, pd):
    data_path = Path("sampled_user_month_traj.parquet")

    df = pd.read_parquet(data_path)
    df["calendarDate"] = pd.to_datetime(df["calendarDate"])

    df
    return (df,)


@app.cell
def _(df):
    # Q1: Proportion of notes, ratings, and requests by monthly activity level.
    q1 = (
        df.groupby("month_role", dropna=False)[["notesWritten", "notesRated", "notesRequested"]]
        .sum()
        .rename(
            columns={
                "notesWritten": "notes",
                "notesRated": "ratings",
                "notesRequested": "requests",
            }
        )
        .reset_index()
    )
    q1["notes_prop"] = q1["notes"] / q1["notes"].sum()
    q1["ratings_prop"] = q1["ratings"] / q1["ratings"].sum()
    q1["requests_prop"] = q1["requests"] / q1["requests"].sum()
    q1 = q1.sort_values("notes", ascending=False)


    q1
    return (q1,)


@app.cell
def _(pd, px, q1, q2):
    q1_long = q1.melt(
        id_vars="month_role",
        value_vars=["notes_prop", "ratings_prop", "requests_prop"],
        var_name="metric",
        value_name="proportion",
    )
    q1_long["group"] = "All"

    # "Successful" series are taken from Q2 helpful outcome proportions.
    q1_success = q2.rename(
        columns={
            "helpful_notes_prop": "notes_prop",
            "helpful_ratings_prop": "ratings_prop",
            "requests_to_helpful_notes_prop": "requests_prop",
        }
    )
    q1_success_long = q1_success.melt(
        id_vars="month_role",
        value_vars=["notes_prop", "ratings_prop", "requests_prop"],
        var_name="metric",
        value_name="proportion",
    )
    q1_success_long["group"] = "Successful"
    q1_plot_df = pd.concat([q1_long, q1_success_long], ignore_index=True)

    metric_labels = {"notes_prop": "Notes", "ratings_prop": "Ratings", "requests_prop": "Requests"}
    q1_plot_df["metric"] = q1_plot_df["metric"].map(metric_labels)

    # Keep legend ordered by the largest contributor in "All"
    role_order = (
        q1_plot_df[q1_plot_df["group"] == "All"]
        .groupby("month_role")["proportion"]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )
    q1_plot_df["month_role"] = pd.Categorical(
        q1_plot_df["month_role"], categories=role_order, ordered=True
    )

    fig_q1 = px.bar(
        q1_plot_df,
        x="metric",
        y="proportion",
        color="month_role",
        facet_col="group",
        barmode="stack",
        title="Q1 Composition by Activity Level (All vs Successful)",
        labels={
            "metric": "",
            "proportion": "Share within metric",
            "month_role": "Activity level",
            "group": "",
        },
    )

    fig_q1.update_yaxes(tickformat=".0%")
    fig_q1.update_xaxes(categoryorder="array", categoryarray=["Notes", "Ratings", "Requests"])
    fig_q1.for_each_annotation(lambda a: a.update(text=a.text.replace("group=", "")))
    fig_q1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notes, ratings, and requests are all concentrated in a small set of activity levels, but each action is dominated by a different role mix. For note creation, the top shares are single_digit_writer (41.8%), single_note_writer (24.1%), double_digit_writer (16.1%), which is writer-heavy. For rating volume, the top shares are double_digit_rater (39.7%), triple_digit_rater (15.8%), single_digit_rater (14.3%), showing rater-heavy concentration. For requests, the top shares are single_post_requestor (34.0%), single_digit_requestor (32.9%), double_digit_requestor (16.7%), which is requestor-heavy. This indicates role specialization: writer, rater, and requestor roles contribute disproportionately to their corresponding activities.
    """)
    return


@app.cell
def _(df):
    # Q2:
    # - proportion of helpful notes by monthly activity level
    # - proportion of helpful ratings on helpful notes by monthly activity level
    # - proportion of requests that ended in helpful notes by monthly activity level
    # Assumptions:
    # - helpful notes: hits
    # - helpful ratings on helpful notes: correctHelpfuls
    # - requests ending in helpful notes: numRequestsResultingInCrh
    q2 = (
        df.groupby("month_role", dropna=False)[
            ["hits", "correctHelpfuls", "numRequestsResultingInCrh"]
        ]
        .sum()
        .rename(
            columns={
                "hits": "helpful_notes",
                "correctHelpfuls": "helpful_ratings_on_helpful_notes",
                "numRequestsResultingInCrh": "requests_to_helpful_notes",
            }
        )
        .reset_index()
    )

    q2["helpful_notes_prop"] = q2["helpful_notes"] / q2["helpful_notes"].sum()
    q2["helpful_ratings_prop"] = (
        q2["helpful_ratings_on_helpful_notes"]
        / q2["helpful_ratings_on_helpful_notes"].sum()
    )
    q2["requests_to_helpful_notes_prop"] = (
        q2["requests_to_helpful_notes"] / q2["requests_to_helpful_notes"].sum()
    )

    q2 = q2.sort_values("helpful_notes", ascending=False)
    q2
    return (q2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Helpful-note outcomes are also concentrated by activity level. For helpful notes, the top shares are single_digit_writer (38.1%), single_note_writer (23.1%), double_digit_writer (21.9%). For helpful ratings on helpful notes, the top shares are double_digit_rater (39.0%), single_digit_rater (24.1%), single_digit_writer (10.5%). For requests that resulted in helpful notes, the top shares are single_post_requestor (36.5%), single_digit_requestor (33.9%), double_digit_requestor (13.5%). These results show that both content quality outcomes and successful requests are disproportionately generated by a limited set of highly engaged roles.
    """)
    return


@app.cell
def _(df, pd):
    # Q3: How long users stayed before attrition by entry timing and entry activity level.
    # Attrition rule:
    # - user is attrited if last active month < global max month in the dataset.
    df_sorted = df.sort_values(["participantId", "calendarDate"])
    max_date = df_sorted["calendarDate"].max()

    entry = (
        df_sorted.groupby("participantId")
        .first()[["calendarDate", "month_role"]]
        .rename(columns={"calendarDate": "entry_date", "month_role": "entry_role"})
    )

    active = df_sorted[df_sorted["activeMonth"] == True]
    last_active = active.groupby("participantId")["calendarDate"].max().rename("last_active_date")

    users = entry.join(last_active, how="left").dropna(subset=["last_active_date"]).copy()
    users["attrited"] = users["last_active_date"] < max_date
    users["tenure_months"] = (
        users["last_active_date"].dt.to_period("M") - users["entry_date"].dt.to_period("M")
    ).apply(lambda x: x.n) + 1

    attrited = users[users["attrited"]].copy()
    attrited["entry_cohort"] = attrited["entry_date"].dt.to_period("M").astype(str)

    q3_by_cohort_role = (
        attrited.groupby(["entry_cohort", "entry_role"])["tenure_months"]
        .agg(["count", "mean", "median"])
        .reset_index()
        .sort_values(["entry_cohort", "mean"])
    )

    q3_by_entry_role = (
        attrited.groupby("entry_role")["tenure_months"]
        .agg(["count", "mean", "median"])
        .reset_index()
        .sort_values("count", ascending=False)
    )

    q3_by_cohort = (
        attrited.groupby("entry_cohort")["tenure_months"]
        .agg(["count", "mean", "median"])
        .reset_index()
        .sort_values("count", ascending=False)
    )

    q3_overall = pd.DataFrame(
        {
            "n_users_with_activity_history": [len(users)],
            "n_attrited_users": [len(attrited)],
            "attrition_share": [len(attrited) / len(users)],
            "mean_tenure_months": [attrited["tenure_months"].mean()],
            "median_tenure_months": [attrited["tenure_months"].median()],
        }
    )


    q3_by_cohort_role
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Attrition duration differs substantially by both entry behavior and entry timing. In this sample, 19,769 of 20,000 users with activity history are classified as attrited under the observation-window rule, with an overall mean/median tenure before attrition of 7.64 / 1 months. Users entering as single-post requestors tend to churn quickly, while users entering with rater/writer activity persist much longer on average; examples include single_post_requestor (mean 2.12, median 1.0 months), single_note_rater (mean 21.40, median 24.0 months), single_digit_requestor (mean 3.67, median 2.0 months), single_digit_rater (mean 20.55, median 23.0 months). Cohort effects are also visible: later cohorts naturally have shorter observed windows, while earlier cohorts can accumulate longer tenures.
    """)
    return


if __name__ == "__main__":
    app.run()
