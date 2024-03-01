import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import matplotlib.patches as mpatches
from typing import *
from dataclasses import dataclass
from warnings import filterwarnings

filterwarnings("ignore")
from math import ceil


# Formatando eixos do matplotlib
def format_spines(ax, right_border=True):
    """
    This function sets up borders from an axis and personalize colors

    Input:
        Axis and a flag for deciding or not to plot the right border
    Returns:
        Plot configuration
    """
    # Setting up colors
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["top"].set_visible(False)
    if right_border:
        ax.spines["right"].set_color("#CCCCCC")
    else:
        ax.spines["right"].set_color("#FFFFFF")
    ax.patch.set_facecolor("#FFFFFF")


def show_stacked_bars(
    data, x_value, y_value, title="Stacked bars plot", type_show="static", width=900
):
    """
      Essa função exibe um gŕafico be barras empilhadas com porcentagem
      utilizando plotly.
      Params:
          data = dataset
          x_value = x axis name
          y_value = y axis name
          title = title for figure
          width = width for figure
    Exemplo de uso:
        show_stacked_bars(data=df,x_value='Previously_Insured', y_value='Response')
    """
    data = (
        pd.crosstab(data[x_value], data[y_value], normalize="index")
        .mul(100)
        .round(2)
        .reset_index()
    )
    data.columns = [
        f"{str(x)}" if str(x).isdigit() else x for x in data.columns.tolist()
    ]
    fig = px.bar(
        data,
        x=x_value,
        y=data.columns.tolist()[1:],
        text="value",
        labels={"variable": f"{y_value}"},
        width=width,
        template="seaborn",
    )
    fig.update_traces(texttemplate="%{text:.2f}%")
    fig.update_xaxes(type="category")
    fig.update_layout(title=title, yaxis_title=f"% {y_value}")
    fig.update_layout(margin = dict(t=50, l=5, r=5, b=5))
    if type_show == "static":
        fig.show("png")
    else:
        fig.show()


def show_grouped_bars(
    ax,
    x_value,
    hue_value,
    dataset,
    title="Bars grouped",
    orient="vertical",
    palette="plasma",
):
    """
     Esta função exibe um gráfico de barras agrupados com a frequência
     e o percentual de cada grupo

    Params:
        ax = axis figure
        x_value = x axis name
        hue_value = hue value name
        dataset
        title
        orient = orientation
        palette
    Exemplo de uso:
        fig, ax = plt.subplots(figsize=(15,7))
        show_grouped_bars(ax, x_value='renda', hue_value='categoria_de_situacao',dataset=student,title='Teste', orient='horizontal')
    """
    if orient == "vertical":
        sns.countplot(x=x_value, hue=hue_value, ax=ax, data=dataset, palette=palette)
        ax.set_title(title, size=12)
    # ax.set_xlabel(x_value, fontsize=12)
    if orient == "horizontal":
        sns.countplot(y=x_value, hue=hue_value, ax=ax, data=dataset, palette=palette)
        ax.set_title(title, size=12)
    # ax.set(xlabel = x_value, ylabel = 'Quantidade e Percentual')
    # y_label = x_value.replace('_',' ')
    # y_label = x_value.replace(x_value[0],x_value[0].upper())
    # ax.set_xlabel(x_value, fontsize=12)
    # ax.set_ylabel("Quantidade e Percentual", fontsize=12)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    labels = [f"{hue_value}={x}" for x in dataset[hue_value].unique().tolist()]
    ax.legend(fontsize=12, labels=labels)
    all_heights = [[p.get_height() for p in bars] for bars in ax.containers]

    # plt.title(title, size=12)
    plt.tight_layout()
    if orient == "vertical":
        for bars in ax.containers:
            for x, p in enumerate(bars):
                total = sum(xgroup[x] for xgroup in all_heights)
                percentage = f"{(100 * p.get_height() / total) :.1f}%"
                ax.annotate(
                    f"{p.get_height()}\n{percentage}",
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    size=12,
                    ha="center",
                    va="bottom",
                )
    ncount = len(dataset)
    if orient == "horizontal":
        n_classes = len(dataset[x_value].unique().tolist())
        for i, p in enumerate(ax.patches):
            x = p.get_bbox().get_points()[1, 0]
            y = p.get_bbox().get_points()[:, 1]
            if (i + n_classes) <= len(ax.patches) - 1:
                ncount = (
                    ax.patches[i].get_bbox().get_points()[1, 0]
                    + ax.patches[i + n_classes].get_bbox().get_points()[1, 0]
                )
            else:
                ncount = (
                    ax.patches[i].get_bbox().get_points()[1, 0]
                    + ax.patches[i - n_classes].get_bbox().get_points()[1, 0]
                )
            # print(p.get_bbox().get_points()[1,0])
            # print(f"{100. * x / ncount}")
            ax.annotate(
                "{} ({:.1f}%)".format(int(x), 100.0 * x / ncount),
                (x, y.mean()),
                va="center",
            )

    x_ticks = [item.get_text() for item in ax.get_xticklabels()]
    if len(x_ticks[0]) > 10:
        ax.set_xticklabels(x_ticks, rotation=45, fontsize=12)
    else:
        ax.set_xticklabels(x_ticks, fontsize=12)


Patch = matplotlib.patches.Patch
PosVal = Tuple[float, Tuple[float, float]]
Axis = matplotlib.axes.Axes
PosValFunc = Callable[[Patch], PosVal]


@dataclass
class AnnotateBars:
    font_size: int = 10
    color: str = "black"
    n_dec: int = 2

    def horizontal(self, ax: Axis, centered=False):
        def get_vals(p: Patch) -> PosVal:
            value = p.get_width()
            div = 2 if centered else 1
            pos = (
                p.get_x() + p.get_width() / div,
                p.get_y() + p.get_height() / 2,
            )
            return value, pos

        ha = "center" if centered else "left"
        self._annotate(ax, get_vals, ha=ha, va="center")

    def vertical(self, ax: Axis, centered: bool = False):
        def get_vals(p: Patch) -> PosVal:
            value = p.get_height()
            div = 2 if centered else 1
            pos = (p.get_x() + p.get_width() / 2, p.get_y() + p.get_height() / div)
            return value, pos

        va = "center" if centered else "bottom"
        self._annotate(ax, get_vals, ha="center", va=va)

    def _annotate(self, ax, func: PosValFunc, **kwargs):
        cfg = {"color": self.color, "fontsize": self.font_size, **kwargs}
        for p in ax.patches:
            value, pos = func(p)
            ax.annotate(f"{value:.{self.n_dec}f}", pos, **cfg)


# Definindo funções úteis para plotagem dos rótulos no gráfico
def make_autopct(values):
    """
    Etapas:
        1. definição de função para formatação dos rótulos

    Argumentos:
        values -- valores extraídos da função value_counts() da coluna de análise [list]

    Retorno:
        my_autopct -- string formatada para plotagem dos rótulos
    """

    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct * total / 100.0))

        return "{p:.1f}%\n({v:d})".format(p=pct, v=val)

    return my_autopct


# Função para plotagem de gráfico de rosca em relação a uma variávei específica do dataset
def donut_plot(
    df,
    col,
    ax,
    label_names=None,
    text="",
    colors=["crimson", "navy"],
    circle_radius=0.8,
    title=f"Gráfico de Rosca",
    flag_ruido=0,
):
    """
    Etapas:
        1. definição de funções úteis para mostrar rótulos em valor absoluto e porcentagem
        2. criação de figura e círculo central de raio pré-definido
        3. plotagem do gráfico de pizza e adição do círculo central
        4. configuração final da plotagem

    Argumentos:
        df -- DataFrame alvo da análise [pandas.DataFrame]
        col -- coluna do DataFrame a ser analisada [string]
        label_names -- nomes customizados a serem plotados como labels [list]
        text -- texto central a ser posicionado [string / default: '']
        colors -- cores das entradas [list / default: ['crimson', 'navy']]
        figsize -- dimensões da plotagem [tupla / default: (8, 8)]
        circle_radius -- raio do círculo central [float / default: 0.8]

    Exemplo de uso:
        fig, ax = plt.subplots()
        label_names = df['Vehicle_Age'].value_counts().index
        donut_plot(df,col='Vehicle_Age',ax=ax,label_names=label_names, colors=['silver','darkviolet','crimson'])
    """

    # Retorno dos valores e definição da figura
    values = df[col].value_counts().values
    if label_names is None:
        label_names = df[col].value_counts().index

    # Verificando parâmetro de supressão de alguma categoria da análise
    if flag_ruido > 0:
        values = values[:-flag_ruido]
        label_names = label_names[:-flag_ruido]

    # Plotando gráfico de rosca
    center_circle = plt.Circle((0, 0), circle_radius, color="white")
    ax.pie(values, labels=label_names, colors=colors, autopct=make_autopct(values))
    ax.add_artist(center_circle)

    # Configurando argumentos do texto central
    kwargs = dict(size=20, fontweight="bold", va="center")
    ax.text(0, 0, text, ha="center", **kwargs)
    ax.set_title(title, size=14, color="dimgrey")


def mean_sum_analysis(
    df, group_col, value_col, orient="vertical", palette="plasma", figsize=(15, 6)
):
    """
    Parâmetros
    ----------
    classifiers: conjunto de classificadores em forma de dicionário [dict]
    X: array com os dados a serem utilizados no treinamento [np.array]
    y: array com o vetor target do modelo [np.array]

    Exemplo de uso: mean_sum_analysis(df_stores[df_stores['StateHoliday']!="RegularDay"], 'StateHoliday','Sales', palette='viridis', figsize=(10, 5))
    """
    # sns.set()
    # Grouping data
    df_mean = df.groupby(group_col, as_index=False)[value_col].mean()
    df_sum = df.groupby(group_col, as_index=False)[value_col].sum()

    # Sorting grouped dataframes
    df_mean.sort_values(by=value_col, ascending=False, inplace=True)
    sorter = list(df_mean[group_col].values)
    sorter_idx = dict(zip(sorter, range(len(sorter))))
    df_sum["mean_rank"] = df_mean[group_col].map(sorter_idx)
    df_sum.sort_values(by="mean_rank", inplace=True)
    df_sum.drop("mean_rank", axis=1, inplace=True)

    # Plotting data
    fig, axs = plt.subplots(ncols=2, figsize=figsize)
    if orient == "vertical":
        sns.barplot(x=value_col, y=group_col, data=df_mean, ax=axs[0], palette=palette)
        sns.barplot(x=value_col, y=group_col, data=df_sum, ax=axs[1], palette=palette)
        AnnotateBars(n_dec=2, font_size=12, color="black").horizontal(axs[0])
        AnnotateBars(n_dec=2, font_size=12, color="black").horizontal(axs[1])
    elif orient == "horizontal":
        sns.barplot(x=group_col, y=value_col, data=df_mean, ax=axs[0], palette=palette)
        sns.barplot(x=group_col, y=value_col, data=df_sum, ax=axs[1], palette=palette)
        AnnotateBars(n_dec=2, font_size=12, color="black").vertical(axs[0])
        AnnotateBars(n_dec=2, font_size=12, color="black").vertical(axs[1])

    # Customizing plot
    for ax in axs:
        format_spines(ax, right_border=False)
        ax.set_ylabel("")
    axs[0].set_title(f"Mean of {value_col} by {group_col}", size=14, color="dimgrey")
    axs[1].set_title(f"Sum of {value_col} by {group_col}", size=14, color="dimgrey")

    plt.tight_layout()
    plt.show()


# Distplot para comparação de densidade das features baseadas na variável target
def distplot(
    df,
    features,
    fig_cols,
    hue=False,
    color=["crimson", "darkslateblue"],
    hist=False,
    figsize=(16, 8),
):
    """
    Etapas:
        1. criação de figura de acordo com as especificações dos argumentos
        2. laço para plotagem de boxplot por eixo
        3. formatação gráfica
        4. validação de eixos excedentes

    Argumentos:
        df -- base de dados para plotagem [pandas.DataFrame]
        features -- conjunto de colunas a serem avaliadas [list]
        fig_cols -- especificações da figura do matplotlib [int]
        hue -- variável resposta contida na base [string -- default: False]
        color_list -- cores para cada classe nos gráficos [list - default: ['crimson', 'darkslateblue']]
        hist -- indicador de plotagem das faixas do histograma [bool - default: False]
        figsize -- dimensões da plotagem [tupla - default: (16, 12)]

    Exemplo de uso:
        distplot(df,features=['Age','Annual_Premium','Vintage'],hue='Response',fig_cols=2,hist=True)
    """

    # Definindo variáveis de controle
    n_features = len(features)
    fig_cols = fig_cols
    fig_rows = ceil(n_features / fig_cols)
    i, j, color_idx = (0, 0, 0)

    # Plotando gráficos
    fig, axs = plt.subplots(nrows=fig_rows, ncols=fig_cols, figsize=figsize)
    # sns.set()
    # Percorrendo por cada uma das features
    for col in features:
        try:
            ax = axs[i, j]
        except:
            ax = axs[j]
        target_idx = 0

        # Plotando, para cada eixo, um gráfico por classe target

        if hue != False:
            labels = [f"{hue}={x}" for x in df[hue].unique().tolist()]
            for classe in df[hue].value_counts().index:
                df_hue = df[df[hue] == classe]
                sns.distplot(df_hue[col], color=color[target_idx], hist=hist, ax=ax)
                ax.legend(labels=labels)
                target_idx += 1
        else:
            sns.distplot(df[col], color=color[0], ax=ax, hist=hist)

        # Incrementando índices
        j += 1
        if j == fig_cols:
            j = 0
            i += 1

        # Customizando plotagem
        ax.set_title(f"Feature: {col}", color="dimgrey", size=14)
        plt.setp(ax, yticks=[])
        # sns.set(style='white')

        sns.despine(left=True)

    # Tratando caso apartado: figura(s) vazia(s)
    i, j = (0, 0)
    for n_plots in range(fig_rows * fig_cols):

        # Se o índice do eixo for maior que a quantidade de features, elimina as bordas
        if n_plots >= n_features:
            try:
                axs[i][j].axis("off")
            except TypeError as e:
                axs[j].axis("off")

        # Incrementando
        j += 1
        if j == fig_cols:
            j = 0
            i += 1

    # Finalizando customização
    plt.tight_layout()
    plt.show()


def boxplot(df, features, fig_cols, hue=False, palette="viridis", figsize=(16, 12)):
    """
    Etapas:
        1. criação de figura de acordo com as especificações dos argumentos
        2. laço para plotagem de boxplot por eixo
        3. formatação gráfica
        4. validação de eixos excedentes

    Argumentos:
        df -- base de dados para plotagem [pandas.DataFrame]
        features -- conjunto de colunas a serem avaliadas [list]
        fig_cols -- especificações da figura do matplotlib [int]
        hue -- variável resposta contida na base [string - default: False]
        palette -- paleta de cores [string / lista - default: 'viridis']
        figsize -- dimensões da figura de plotagem [tupla - default: (16, 12)]

    Exemplo de uso:
        boxplot(df,features=['Age','Annual_Premium','Vintage'],fig_cols=2,figsize=(12,5))
    """

    # Definindo variáveis de controle
    n_features = len(features)
    fig_rows = ceil(n_features / fig_cols)
    i, j, color_idx = (0, 0, 0)

    # Plotando gráficos
    fig, axs = plt.subplots(nrows=fig_rows, ncols=fig_cols, figsize=figsize)

    # Plotando gráfico
    for col in features:
        try:
            ax = axs[i, j]
        except:
            ax = axs[j]

        # Plotando gráfico atribuindo a variável target como hue
        if hue != False:
            # labels = [ f'{hue}={x}' for x in df[hue].unique().tolist()]
            sns.boxplot(x=df[hue], y=df[col], ax=ax, palette=palette)
            # ax.legend(labels=labels)
        else:
            sns.boxplot(y=df[col], ax=ax, palette=palette)

        # Formatando gráfico
        format_spines(ax, right_border=False)
        ax.set_title(f"Feature: {col.upper()}", size=14, color="dimgrey")
        plt.tight_layout()

        # Incrementando índices
        j += 1
        if j == fig_cols:
            j = 0
            i += 1

    # Tratando caso apartado: figura(s) vazia(s)
    i, j = (0, 0)
    for n_plots in range(fig_rows * fig_cols):

        # Se o índice do eixo for maior que a quantidade de features, elimina as bordas
        if n_plots >= n_features:
            try:
                axs[i][j].axis("off")
            except TypeError as e:
                axs[j].axis("off")

        # Incrementando
        j += 1
        if j == fig_cols:
            j = 0
            i += 1


# Função para plotagem de volumetria das variáveis categóricas do conjunto de dados
def catplot_analysis(
    df_categorical, fig_cols=3, hue=False, palette="viridis", figsize=(16, 10)
):
    """
    Etapas:
        1. retorno das variáveis categóricas do conjunto de dados
        2. parametrização de variáveis de plotagem
        3. aplicação de laços de repetição para plotagens / formatação

    Argumentos:
        df -- conjunto de dados a ser analisado [pandas.DataFrame]
        fig_cols -- quantidade de colunas da figura matplotlib [int]

    Exemplo de uso:
        catplot_analysis( df_stores.select_dtypes(include='object').drop(['Date','YearWeek','CompetitionSince','PromoSince'],axis=1))
    """

    # Retornando parâmetros para organização da figura
    if hue != False:
        cat_features = list(df_categorical.drop(hue, axis=1).columns)
    else:
        cat_features = list(df_categorical.columns)

    total_cols = len(cat_features)
    fig_cols = fig_cols
    fig_rows = ceil(total_cols / fig_cols)
    ncount = len(cat_features)

    # Retornando parâmetros para organização da figura
    # sns.set(style='white', palette='muted', color_codes=True)
    total_cols = len(cat_features)
    fig_rows = ceil(total_cols / fig_cols)

    # Criando figura de plotagem
    fig, axs = plt.subplots(nrows=fig_rows, ncols=fig_cols, figsize=(figsize))
    i, j = 0, 0

    # Laço de repetição para plotagem categórica
    for col in cat_features:
        # Indexando variáveis e plotando gráfico
        try:
            ax = axs[i, j]
        except:
            ax = axs[j]
        if hue != False:
            sns.countplot(
                y=col,
                data=df_categorical,
                palette=palette,
                ax=ax,
                hue=hue,
                order=df_categorical[col].value_counts().index,
            )
        else:
            sns.countplot(
                y=col,
                data=df_categorical,
                palette=palette,
                ax=ax,
                order=df_categorical[col].value_counts().index,
            )

        # Customizando gráfico
        format_spines(ax, right_border=False)
        AnnotateBars(n_dec=0, color="dimgrey").horizontal(ax)
        ax.set_title(col)

        # Incrementando índices de eixo
        j += 1
        if j == fig_cols:
            j = 0
            i += 1

    # Tratando caso apartado: figura(s) vazia(s)
    i, j = (0, 0)
    for n_plots in range(fig_rows * fig_cols):

        # Se o índice do eixo for maior que a quantidade de features, elimina as bordas
        if n_plots >= len(cat_features):
            try:
                axs[i][j].axis("off")
            except TypeError as e:
                axs[j].axis("off")

        # Incrementando
        j += 1
        if j == fig_cols:
            j = 0
            i += 1

    plt.tight_layout()
    plt.show()


def data_overview(
    df,
    corr=False,
    label_name=None,
    sort_by="qtd_null",
    thresh_percent_null=0,
    thresh_corr_label=0,
):
    """
    Etapas:
        1. levantamento de atributos com dados nulos no conjunto
        2. análise do tipo primitivo de cada atributo
        3. análise da quantidade de entradas em caso de atributos categóricos
        4. extração da correlação pearson com o target para cada atributo
        5. aplicação de regras definidas nos argumentos
        6. retorno do dataset de overview criado

    Argumentos:
        df -- DataFrame a ser analisado [pandas.DataFrame]
        label_name -- nome da variável target [string]
        sort_by -- coluna de ordenação do dataset de overview [string - default: 'qtd_null']
        thresh_percent_null -- filtro de dados nulos [int - default: 0]
        threh_corr_label -- filtro de correlação com o target [int - default: 0]

    Exemplo de uso:
        data_overview(df,corr=True,label_name='Response')
    """

    # Criando DataFrame com informações de dados nulos
    df_null = pd.DataFrame(df.isnull().sum()).reset_index()
    df_null.columns = ["feature", "qtd_null"]
    df_null["percent_null"] = df_null["qtd_null"] / len(df)

    # Retornando tipo primitivo e qtd de entradas para os categóricos
    df_null["dtype"] = df_null["feature"].apply(lambda x: df[x].dtype)
    df_null["qtd_cat"] = [
        len(df[col].value_counts()) if df[col].dtype == "object" else 0
        for col in df_null["feature"].values
    ]

    if corr:
        # Extraindo informação de correlação com o target
        label_corr = pd.DataFrame(df.corr()[label_name])
        label_corr = label_corr.reset_index()
        label_corr.columns = ["feature", "target_pearson_corr"]

        # Unindo informações
        df_null_overview = df_null.merge(label_corr, how="left", on="feature")
        df_null_overview.query("target_pearson_corr > @thresh_corr_label")
    else:
        df_null_overview = df_null

    # Filtrando dados nulos de acordo com limiares
    df_null_overview.query("percent_null > @thresh_percent_null")

    # Ordenando DataFrame
    df_null_overview = df_null_overview.sort_values(by=sort_by, ascending=False)
    df_null_overview = df_null_overview.reset_index(drop=True)

    return df_null_overview


def box_and_kde_analysis(
    df, target, hue, palette="plasma", fig_cols=2, figsize=(12, 5)
):
    """
    df: dataset
    target:variável resposta
    hue: array de variáveis categóricas
    fig_cols:número de colunas

    Exemplo de uso:
        box_and_kde_analysis(df_stores,target='Sales', fig_cols=2, hue=['Assortment','StoreType','StateHoliday'], figsize=(12,8))
    """

    n_features = len(hue)
    # fig_rows = ceil(n_features / fig_cols)
    i, j, color_idx = (0, 0, 0)

    # Plotando gráficos
    fig, axs = plt.subplots(nrows=n_features, ncols=fig_cols, figsize=figsize)

    for feature in hue:
        ax = axs[i, j]
        sns.boxplot(df, x=feature, y=target, ax=ax)
        format_spines(ax)
        i += 1

        ax.set_title(f"Feature: {feature.upper()}", size=14, color="dimgrey")

    j = 1
    i = 0
    for feature in hue:
        ax = axs[i, j]
        i += 1
        sns.kdeplot(df, x=target, hue=feature, ax=ax, fill=True)
        format_spines(ax)
        ax.set_title(f"Feature: {feature.upper()}", size=14, color="dimgrey")
    plt.tight_layout()


# Função responsável por plotar volumetria de uma única variável categórica em formato atualizado
def single_countplot(
    df,
    ax,
    x=None,
    y=None,
    top=None,
    order=True,
    hue=False,
    palette="plasma",
    width=0.75,
    sub_width=0.3,
    sub_size=12,
):
    """
    Parâmetros
    ----------
    classifiers: conjunto de classificadores em forma de dicionário [dict]
    X: array com os dados a serem utilizados no treinamento [np.array]
    y: array com o vetor target do modelo [np.array]

    Exemplo de uso:
        fig, ax = plt.subplots(figsize=(14, 6))
        single_countplot(df, y='Vehicle_Age', ax=ax)
    """
    # sns.set('dark')
    # Verificando plotagem por quebra de alguma variável categórica
    ncount = len(df)
    if x:
        col = x
    else:
        col = y

    # Verificando a plotagem de top categorias
    if top is not None:
        cat_count = df[col].value_counts()
        top_categories = cat_count[:top].index
        df = df[df[col].isin(top_categories)]

    # Validando demais argumentos e plotando gráfico
    if hue != False:
        if order:
            sns.countplot(
                x=x,
                y=y,
                data=df,
                palette=palette,
                ax=ax,
                order=df[col].value_counts().index,
                hue=hue,
                width=width,
            )
        else:
            sns.countplot(
                x=x, y=y, data=df, palette=palette, ax=ax, hue=hue, width=width
            )
    else:
        if order:
            sns.countplot(
                x=x,
                y=y,
                data=df,
                palette=palette,
                ax=ax,
                width=width,
                order=df[col].value_counts().index,
            )
        else:
            sns.countplot(x=x, y=y, data=df, palette=palette, ax=ax, width=width)

    # Formatando eixos
    format_spines(ax, right_border=False)
    # print('teste')

    # Inserindo rótulo de percentual
    if x:
        for p in ax.patches:
            x = p.get_bbox().get_points()[:, 0]
            y = p.get_bbox().get_points()[1, 1]
            ax.annotate(
                "{}\n{:.1f}%".format(int(y), 100.0 * y / ncount),
                (x.mean(), y),
                ha="center",
                va="bottom",
            )
    else:
        for p in ax.patches:
            x = p.get_bbox().get_points()[1, 0]
            y = p.get_bbox().get_points()[:, 1]
            ax.annotate(
                "{} ({:.1f}%)".format(int(x), 100.0 * x / ncount),
                (x, y.mean()),
                va="center",
            )


# Função para análise da matriz de correlação
def target_correlation_matrix(
    data,
    label_name,
    ax,
    n_vars=10,
    corr="positive",
    fmt=".2f",
    cmap="YlGnBu",
    cbar=True,
    annot=True,
    square=True,
):
    """
    Etapas:
        1. construção de correlação entre as variáveis
        2. filtragem das top k variáveis com maior correlação
        3. plotagem e configuração da matriz de correlação

    Argumentos:
        data -- DataFrame a ser analisado [pandas.DataFrame]
        label_name -- nome da coluna contendo a variável resposta [string]
        n_vars -- indicador das top k variáveis a serem analisadas [int]
        corr -- indicador booleano para plotagem de correlações ('positive', 'negative') [string]
        fmt -- formato dos números de correlação na plotagem [string]
        cmap -- color mapping [string]
        figsize -- dimensões da plotagem gráfica [tupla]
        cbar -- indicador de plotagem da barra indicadora lateral [bool]
        annot -- indicador de anotação dos números de correlação na matriz [bool]
        square -- indicador para redimensionamento quadrático da matriz [bool]

    Exemplo de uso:
        fig, ax = plt.subplots(figsize=(12,8))
        target_correlation_matrix(df,'Response',ax,cbar=False,cmap='inferno')
    """

    # Criando matriz de correlação para a base de dados
    corr_mx = data.corr()

    # Retornando apenas as top k variáveis com maior correlação frente a variável resposta
    if corr == "positive":
        corr_cols = list(corr_mx.nlargest(n_vars + 1, label_name)[label_name].index)
        title = f"Top {n_vars} Features - Correlação Positiva com o Target"
    elif corr == "negative":
        corr_cols = list(corr_mx.nsmallest(n_vars + 1, label_name)[label_name].index)
        corr_cols = [label_name] + corr_cols[:-1]
        title = f"Top {n_vars} Features - Correlação Negativa com o Target"
        cmap = "magma"

    corr_data = np.corrcoef(data[corr_cols].values.T)

    # Construindo plotagem da matriz
    sns.heatmap(
        corr_data,
        ax=ax,
        cbar=cbar,
        annot=annot,
        square=square,
        fmt=fmt,
        cmap=cmap,
        yticklabels=corr_cols,
        xticklabels=corr_cols,
    )
    ax.set_title(title, size=14, color="dimgrey", pad=20)

    return


def mean_or_sum_help(
    df,
    ax,
    group_col,
    value_col,
    orient="vertical",
    hue=None,
    palette="plasma",
    top=None,
    mode="mean",
    title=None,
):
    if orient == "vertical":
        sns.barplot(
            x=group_col, y=value_col, data=df.head(top), ax=ax, hue=hue, palette=palette
        )
        AnnotateBars(n_dec=2, font_size=12, color="black").vertical(ax)
        ax.set_ylabel("")
       # if mode == "mean":
        ax.set_title(title, size=14, color="dimgrey")
        format_spines(ax)
        # else:
        #     ax.set_title(title, size=14, color="dimgrey")
        #     format_spines(ax)

    elif orient == "horizontal":
        if type(df[group_col].values[0]) != str:
            df[group_col] = df[group_col].astype(str)
        sns.barplot(
            y=group_col, x=value_col, data=df.head(top), hue=hue, ax=ax, palette=palette
        )
        AnnotateBars(n_dec=2, font_size=12, color="black").horizontal(ax)
        ax.set_ylabel("")
       # if mode == "mean":
        ax.set_title(title, size=14, color="dimgrey")
        format_spines(ax)
       # else:
        #ax.set_title(title, size=14, color="dimgrey")
        #format_spines(ax)
        
        # if mode == "mean":
        #     ax.set_title(
        #        f"Mean of {value_col} by {group_col} and {hue}",
        #         title,
        #         size=14,
        #         color="dimgrey",
        #     )
        #     format_spines(ax, right_border=False)
        # else:
        #     ax.set_title(
        #         f"Sum of {value_col} by {group_col} and {hue}", size=14, color="dimgrey"
        #     )
        #     format_spines(ax, right_border=False)


def mean_or_sum_analysis(
    df,
    ax,
    group_col,
    value_col,
    orient="vertical",
    hue=None,
    palette="plasma",
    top=None,
    mode="mean",
    figsize=(15, 6),
):
    """
    Parâmetros
    ----------
    classifiers: conjunto de classificadores em forma de dicionário [dict]
    X: array com os dados a serem utilizados no treinamento [np.array]
    y: array com o vetor target do modelo [np.array]

    Exemplo de uso:
    fig, ax = plt.subplots(figsize=(18,6 ))
    mean_or_sum_analysis(df_stores,ax=ax,
              group_col='Month',value_col='Sales', palette='Set1', orient='vertical',mode='sum', hue='SchoolHoliday')
    """
    # sns.set()
    # Grouping data

    if mode == "mean":
        if hue != None:
            title = f"Mean of {value_col} by {group_col} and {hue}"
            df_mean = (
                df[[group_col, hue, value_col]]
                .groupby([group_col, hue])
                .mean()
                .reset_index()
            )
            mean_or_sum_help(
                df=df_mean,
                ax=ax,
                group_col=group_col,
                value_col=value_col,
                orient=orient,
                mode=mode,
                hue=hue,
                palette=palette,
                title=title,
            )
        else:
            title = f"Mean of {value_col} by {group_col}"
            df_mean = df.groupby(group_col, as_index=False)[value_col].mean()
            df_mean.sort_values(by=value_col, ascending=False, inplace=True)
            mean_or_sum_help(
                df=df_mean,
                ax=ax,
                group_col=group_col,
                value_col=value_col,
                orient=orient,
                mode=mode,
                hue=hue,
                top=top,
                palette=palette,
                title=title,
            )
    if mode == "sum":
        if hue != None:
            title = f"Sum of {value_col} by {group_col} and {hue}"
            df_sum = (
                df[[group_col, hue, value_col]]
                .groupby([group_col, hue])
                .sum()
                .reset_index()
            )
            mean_or_sum_help(
                df=df_sum,
                ax=ax,
                group_col=group_col,
                value_col=value_col,
                orient=orient,
                mode=mode,
                hue=hue,
                palette=palette,
                title=title,
            )

        else:
            title = f"Sum of {value_col} by {group_col}"
            df_sum = df.groupby(group_col, as_index=False)[value_col].sum()
            df_sum.sort_values(by=value_col, ascending=False,inplace=True)
            mean_or_sum_help(
                df=df_sum,
                ax=ax,
                group_col=group_col,
                value_col=value_col,
                orient=orient,
                mode=mode,
                hue=hue,
                top=top,
                palette=palette,
                title=title,
            )
    #plt.tight_layout()
    #plt.show()


# def mean_or_sum_analysis(df,ax, group_col, value_col, orient='vertical', palette='plasma',top=None,mode='mean',figsize=(15, 6)):
#     """
#     Parâmetros
#     ----------
#     classifiers: conjunto de classificadores em forma de dicionário [dict]
#     X: array com os dados a serem utilizados no treinamento [np.array]
#     y: array com o vetor target do modelo [np.array]

#     Exemplo de uso:
#         fig,ax = plt.subplots()
#         mean_analysis(df_stores,ax, group_col='StateHoliday',value_col='Sales',
#         top=10,palette='viridis', figsize=(10, 5))
#     """
#     #sns.set()
#     # Grouping data

#     if mode=='mean':
#         df_mean = df.groupby(group_col, as_index=False).mean()
#     #df_sum = df.groupby(group_col, as_index=False).sum()

#     # Sorting grouped dataframes
#         df_mean.sort_values(by=value_col, ascending=False, inplace=True)
#         sorter = list(df_mean[group_col].values)
#         sorter_idx = dict(zip(sorter, range(len(sorter))))
#         if orient == 'vertical':
#                 sns.barplot(x=group_col, y=value_col, data=df_mean.head(top), ax=ax, palette=palette)
#                 AnnotateBars(n_dec=2, font_size=12, color='black').vertical(ax)
#                 format_spines(ax, right_border=True)
#         elif orient == 'horizontal':
#                 sns.barplot(y=group_col, x=value_col, data=df_mean.head(top), ax=ax, palette=palette)
#                 AnnotateBars(n_dec=2, font_size=12, color='black').horizontal(ax)
#                 format_spines(ax, right_border=False)

#         ax.set_ylabel('')
#         ax.set_title(f'Mean of {value_col} by {group_col}', size=14, color='dimgrey')
#     if mode=='sum':
#         df_sum = df.groupby(group_col, as_index=False).sum()
#         df_sum.sort_values(by=value_col,ascending=False)

#         if orient == 'vertical':
#                 sns.barplot(x=group_col, y=value_col, data=df_sum.head(top), ax=ax, palette=palette)
#                 AnnotateBars(n_dec=2, font_size=12, color='black').vertical(ax)
#                 format_spines(ax, right_border=True)

#         elif orient == 'horizontal':
#                 sns.barplot(y=group_col, x=value_col, data=df_sum.head(top), ax=ax, palette=palette)
#                 AnnotateBars(n_dec=2, font_size=12, color='black').horizontal(ax)
#                 format_spines(ax, right_border=False)
#         ax.set_ylabel('')
#         ax.set_title(f'Sum of {value_col} by {group_col}', size=14, color='dimgrey')

#     plt.tight_layout()
#     plt.show()
