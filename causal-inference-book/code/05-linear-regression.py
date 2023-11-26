import pathlib
import logging
import pandas as pd
import statsmodels.formula.api as smf


SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


def get_csv_data(csv_file):
    return pd.read_csv(csv_file)


def exam_online_regression(data):
    result = smf.ols("falsexam ~ format_ol", data=data).fit()
    log.debug("\n" + str(result.summary().tables[1]))
    falsexam_means = data.groupby("format_ol")["falsexam"].mean()
    log.debug(
        "mean for format_ol=0: {:.4f} and format_ol=1: {:.4f}".format(
            falsexam_means[0], falsexam_means[1]
        )
    )
    log.debug(
        "difference between means {:.4f}".format(
            falsexam_means[1] - falsexam_means[0]
        )
    )
    return result


def wage_education_short_regression(data):
    result = smf.ols("lhwage ~ educ", data=data).fit()
    log.debug("\n" + str(result.summary().tables[1]))
    educ_coef = result.summary2().tables[1]["Coef."].educ
    log.debug(f"Coefficient for education is {educ_coef:.4f}")


def wage_education_long_regression(data):
    controls = [
        "IQ",
        "exper",
        "tenure",
        "age",
        "married",
        "black",
        "south",
        "urban",
        "sibs",
        "brthord",
        "meduc",
        "feduc",
    ]
    result = smf.ols("lhwage ~ educ +" + "+".join(controls), data=data).fit()
    log.debug("\n" + str(result.summary().tables[1]))
    educ_coef = result.summary2().tables[1]["Coef."].educ
    log.debug(f"Coefficient for education is {educ_coef:.4f}")


def main():
    DATA_DIR = SCRIPT_DIR / ".." / "data"

    file_name = "online_classroom.csv"
    log.info(f"Get data for {file_name}")
    raw_data = get_csv_data(DATA_DIR / file_name)
    data = raw_data[raw_data.format_blended == 0]
    _ = exam_online_regression(data)

    file_name = "wage.csv"
    log.info(f"Get data for {file_name}")
    data = get_csv_data(DATA_DIR / file_name)
    _ = wage_education_short_regression(data)
    _ = wage_education_long_regression(data)


if __name__ == "__main__":
    main()
