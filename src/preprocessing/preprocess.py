from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(X):
    """
    Construye el preprocesador para variables
    numéricas y categóricas.
    """

    numerical_columns = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                numerical_columns,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_columns,
            ),
        ]
    )

    return preprocessor
