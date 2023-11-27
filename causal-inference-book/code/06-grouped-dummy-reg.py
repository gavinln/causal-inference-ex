import pathlib
import logging
import pandas as pd
import statsmodels.formula.api as smf


SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


def get_csv_data(csv_file):
    return pd.read_csv(csv_file)


def wage_education_regression(data):
    log.debug("wage education regression")
    result = smf.ols("lhwage ~ educ", data=data).fit()
    log.debug("\n" + str(result.summary().tables[1]))


def wage_education_weighted_regression(data):
    log.debug("wage education weighted regression")

    group_wage = (
        data.assign(count=1)
        .groupby("educ")
        .agg({"lhwage": "mean", "count": "count"})
        .reset_index()
    )

    result = smf.wls(
        "lhwage ~ educ", data=group_wage, weights=group_wage["count"]
    ).fit()
    log.debug("\n" + str(result.summary().tables[1]))


def wage_education_iq_weighted_regression(data):
    log.debug("wage education iq weighted regression")

    group_wage = (
        data.assign(count=1)
        .groupby("educ")
        .agg({"lhwage": "mean", "IQ": "mean", "count": "count"})
        .reset_index()
    )

    result = smf.wls(
        "lhwage ~ educ + IQ", data=group_wage, weights=group_wage["count"]
    ).fit()
    log.debug("\n" + str(result.summary().tables[1]))


def wage_educ_dummy_reg(data):
    wage = data.assign(hwage=lambda d: d["wage"] / d["hours"]).assign(
        T=lambda d: (d["educ"] > 12).astype(int)
    )

    log.debug("wage education dummy regression")
    result = smf.ols("hwage ~ T", data=wage).fit()
    log.debug("\n" + str(result.summary().tables[1]))


def wage_educ_dummy_iq_reg(data):
    wage = data.assign(hwage=lambda d: d["wage"] / d["hours"]).assign(
        T=lambda d: (d["educ"] > 12).astype(int)
    )

    log.debug("wage education dummy IQ regression")
    result = smf.ols("hwage ~ T + IQ", data=wage).fit()
    log.debug("\n" + str(result.summary().tables[1]))


def wage_educ_dummy_iq_interact_reg(data):
    wage = data.assign(hwage=lambda d: d["wage"] / d["hours"]).assign(
        T=lambda d: (d["educ"] > 12).astype(int)
    )

    log.debug("wage education dummy IQ interaction regression")
    result = smf.ols("hwage ~ T * IQ", data=wage).fit()
    log.debug("\n" + str(result.summary().tables[1]))


def wage_educ_categorical_reg(data):
    wage = data.assign(hwage=lambda d: d["wage"] / d["hours"])

    log.debug("wage education categorical regression")
    result = smf.ols("hwage ~ C(educ)", data=wage).fit()
    log.debug("\n" + str(result.summary().tables[1]))


def wage_educ_categorical_iq_bins_reg(data):
    wage = data.assign(hwage=lambda d: d["wage"] / d["hours"]).assign(
        IQ_bins = lambda d: pd.qcut(d["IQ"], q=4, labels=range(4)))

    log.debug("wage education categorical IQ bins regression")
    result = smf.ols("hwage ~ C(educ) + C(IQ_bins)", data=wage).fit()
    log.debug("\n" + str(result.summary().tables[1]))


def main():
    DATA_DIR = SCRIPT_DIR / ".." / "data"

    file_name = "wage.csv"
    log.info(f"Get data for {file_name}")
    raw_data = get_csv_data(DATA_DIR / file_name)
    data = raw_data[["wage", "lhwage", "educ", "IQ"]]

    wage_education_regression(data)

    wage_education_weighted_regression(data)

    wage_education_iq_weighted_regression(data)

    wage_educ_dummy_reg(raw_data)

    wage_educ_dummy_iq_reg(raw_data)

    wage_educ_dummy_iq_interact_reg(raw_data)

    wage_educ_categorical_reg(raw_data)

    wage_educ_categorical_iq_bins_reg(raw_data)


if __name__ == "__main__":
    main()
