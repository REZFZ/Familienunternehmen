import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Daten laden
data = pd.read_csv('Data.csv')
plt.rcParams.update({'font.size': 12, 'font.family': 'Arial'})
data['Q3_Familienbesitz'] = data['Q3_Familienbesitz'].astype('category')
for col in data.columns[3:]:
    data[col] = pd.to_numeric(data[col], errors='coerce')
# Daten für die Visualisierung vorbereiten
melted_data = pd.melt(data, id_vars=['Q3_Familienbesitz'], value_vars=data.columns[3:],
                      var_name='Frage', value_name='Antwort')
# Fragenüberschriften
question_titles = {
    'Q4_WerteIntegration': 'Integration der Unternehmenswerte',
    'Q5_WerteEffektivitaet': 'Effektivität der Integration',
    'Q6_EthikReviewHaeufigkeit': 'Häufigkeit der Überprüfung',
    'Q7_ComplianceEinhaltung': 'Einhaltung der Compliance',
    'Q8_PersoenlicherEinfluss': 'Einfluss persönlicher Beziehungen',
    'Q9_Konflikthaeufigkeit': 'Häufigkeit von Interessenkonflikten',
    'Q10_Konfliktmanagement': 'Effektivität des Konfliktmanagements',
    'Q11_Transparenz': 'Transparenz der Prozesse',
    'Q12_RichtlinienFormalisierung': 'Formalisierung der Richtlinien',
    'Q13_RichtlinienDurchsetzungsEffekt': 'Durchsetzungseffektivität',
    'Q14_RichtlinienUpdateHaeufigkeit': 'Häufigkeit der Updates',
    'Q15_Mitarbeiterbeteiligung': 'Beteiligung der Mitarbeiter'
}
answer_labels = {
    'Q4_WerteIntegration': {2: 'Schwach ', 5: 'Vollständig '},
    'Q5_WerteEffektivitaet': {2: 'Wenig ', 5: 'Hoch'},
    'Q6_EthikReviewHaeufigkeit': {2: 'Selten', 5: 'Immer'},
    'Q7_ComplianceEinhaltung': {2: 'Schlecht', 5: 'Ausgezeichnet'},
    'Q8_PersoenlicherEinfluss': {2: 'Geringfügig', 5: 'Überwiegend'},
    'Q9_Konflikthaeufigkeit': {2: 'Selten', 5: 'Immer'},
    'Q10_Konfliktmanagement': {2: 'Ineffektiv', 5: 'Effektiv'},
    'Q11_Transparenz': {2: 'Undurchsichtig', 5: 'transparent'},
    'Q12_RichtlinienFormalisierung': {2: 'Schwach ', 5: 'Vollständig '},
    'Q13_RichtlinienDurchsetzungsEffekt': {2: 'Leicht ', 5: 'Positiv'},
    'Q14_RichtlinienUpdateHaeufigkeit': {2: 'Selten', 5: 'Kontinuierlich'},
    'Q15_Mitarbeiterbeteiligung': {2: 'Wenig ', 5: ' Stark '}
}

group1 = ['Q4_WerteIntegration', 'Q5_WerteEffektivitaet', 'Q6_EthikReviewHaeufigkeit', 'Q7_ComplianceEinhaltung']
group2 = ['Q8_PersoenlicherEinfluss', 'Q9_Konflikthaeufigkeit', 'Q10_Konfliktmanagement', 'Q11_Transparenz']
group3 = ['Q12_RichtlinienFormalisierung', 'Q13_RichtlinienDurchsetzungsEffekt', 'Q14_RichtlinienUpdateHaeufigkeit',
          'Q15_Mitarbeiterbeteiligung']


def plot_group(data, group, file_name, dpi=200):
    melted_data_group = data[data['Frage'].isin(group)]
    g = sns.FacetGrid(melted_data_group, col='Frage', col_wrap=2, sharey=False, height=4)
    g.map_dataframe(sns.violinplot, x='Q3_Familienbesitz', y='Antwort', hue='Q3_Familienbesitz', palette='muted',
                    legend=False)

    for ax in g.axes.flat:
        question = ax.get_title().replace('Frage = ', '')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Familienunternehmen', 'Nicht-Familienunternehmen'])
        ax.set_yticks([1, 2, 3, 4, 5])

        if question in answer_labels:
            labels = [''] * 5
            for key, value in answer_labels[question].items():
                labels[key - 1] = value
            ax.set_yticklabels(labels)
        else:
            ax.set_yticklabels([1, 2, 3, 4, 5])

        if question in question_titles:
            ax.set_title(question_titles[question])

    plt.subplots_adjust(top=0.9)
    g.fig.suptitle(f' {file_name}')

# Erster Plot
plot_group(melted_data, group1, 'Violin Plots, Fragen 4 bis 7')

# Zweiter Plot
plot_group(melted_data, group2, 'Violin Plots, Fragen 8 bis 11')

# Dritter Plot
plot_group(melted_data, group3, 'Violin Plots, Fragen 12 bis 15')

##########################################################
import pandas as pd
from scipy.stats import mannwhitneyu

data = pd.read_csv('Data.csv')
print(data['Q3_Familienbesitz'].unique())
data['Q3_Familienbesitz'] = data['Q3_Familienbesitz'].replace({2: 0})
data['Q3_Familienbesitz'] = data['Q3_Familienbesitz'].astype('category')
print(data['Q3_Familienbesitz'].value_counts())
def mann_whitney_u_test(data, dependent_vars, title):
    results = {}

    for var in dependent_vars:
        family_owned = data[data['Q3_Familienbesitz'] == 1][var].dropna()
        non_family_owned = data[data['Q3_Familienbesitz'] == 0][var].dropna()

        # Debug-Ausgaben zur Überprüfung der Größe der Gruppen
        print(f'Anzahl der Familienunternehmen für {var}: {len(family_owned)}')
        print(f'Anzahl der Nicht-Familienunternehmen für {var}: {len(non_family_owned)}')

        if len(family_owned) > 0 and len(non_family_owned) > 0:
            try:
                stat, p = mannwhitneyu(family_owned, non_family_owned, alternative='two-sided')
                results[var] = (stat, p)
            except Exception as e:
                print(f"Fehler bei der Durchführung des Mann-Whitney-U-Tests für {var}: {e}")
        else:
            print(f'Nicht genügend Daten für den Mann-Whitney-U-Test für {var}')

    print(f'Mann-Whitney-U-Testergebnisse für {title}:, dpi=200')
    for key in results:
        print(f'{key}: U-Statistik = {results[key][0]}, p-Wert = {results[key][1]}')
        print('\n')
# Hypothese 1: Q4 bis Q7
mann_whitney_u_test(data, ['Q4_WerteIntegration', 'Q5_WerteEffektivitaet', 'Q6_EthikReviewHaeufigkeit', 'Q7_ComplianceEinhaltung'], 'Hypothese 1')
# Hypothese 2: Q8 bis Q11
mann_whitney_u_test(data, ['Q8_PersoenlicherEinfluss', 'Q9_Konflikthaeufigkeit', 'Q10_Konfliktmanagement', 'Q11_Transparenz'], 'Hypothese 2')
# Hypothese 3: Q12 bis Q15
mann_whitney_u_test(data, ['Q12_RichtlinienFormalisierung', 'Q13_RichtlinienDurchsetzungsEffekt', 'Q14_RichtlinienUpdateHaeufigkeit', 'Q15_Mitarbeiterbeteiligung'], 'Hypothese 3')
###############################
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
data = pd.read_csv('Data.csv')
data['Q3_Familienbesitz'] = data['Q3_Familienbesitz'].astype('category')
# Define function for Mann-Whitney U Test
def mann_whitney_u_test(group1, group2):
    stat, p = mannwhitneyu(group1, group2)
    return stat, p
def perform_mann_whitney(data, questions):
    results = {'Frage': [], 'U-Statistik': [], 'p-Wert': []}
    for question in questions:
        group1 = data[data['Q3_Familienbesitz'] == 1][question]
        group2 = data[data['Q3_Familienbesitz'] == 2][question]
        if len(group1) > 0 and len(group2) > 0:
            stat, p = mann_whitney_u_test(group1, group2)
            results['Frage'].append(question)
            results['U-Statistik'].append(stat)
            results['p-Wert'].append(p)
    return results
# Define questions for each hypothesis
questions_h1 = ['Q4_WerteIntegration', 'Q5_WerteEffektivitaet', 'Q6_EthikReviewHaeufigkeit', 'Q7_ComplianceEinhaltung']
questions_h2 = ['Q8_PersoenlicherEinfluss', 'Q9_Konflikthaeufigkeit', 'Q10_Konfliktmanagement', 'Q11_Transparenz']
questions_h3 = ['Q12_RichtlinienFormalisierung', 'Q13_RichtlinienDurchsetzungsEffekt', 'Q14_RichtlinienUpdateHaeufigkeit', 'Q15_Mitarbeiterbeteiligung']
# Get results for each hypothesis
results_h1 = perform_mann_whitney(data, questions_h1)
results_h2 = perform_mann_whitney(data, questions_h2)
results_h3 = perform_mann_whitney(data, questions_h3)
# Convert results to DataFrame
df_h1 = pd.DataFrame(results_h1)
df_h2 = pd.DataFrame(results_h2)
df_h3 = pd.DataFrame(results_h3)
# Plot the results for Hypothesis 1
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x='Frage', y='U-Statistik', data=df_h1, hue='Frage', palette='Blues_d', dodge=False, ax=ax, legend=False)
for index, row in df_h1.iterrows():
    ax.text(row.name, row['U-Statistik'] + 10, f'p={row["p-Wert"]:.3e}', color='black', ha="center")  # Adjust the y position
ax.set_title('Mann-Whitney-U-Testergebnisse für Hypothese 1')
# Plot the results for Hypothesis 2
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x='Frage', y='U-Statistik', data=df_h2, hue='Frage', palette='Greens_d', dodge=False, ax=ax, legend=False)
for index, row in df_h2.iterrows():
    ax.text(row.name, row['U-Statistik'] + 10, f'p={row["p-Wert"]:.3e}', color='black', ha="center")  # Adjust the y position
ax.set_title('Mann-Whitney-U-Testergebnisse für Hypothese 2')
# Plot the results for Hypothesis 3
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x='Frage', y='U-Statistik', data=df_h3, hue='Frage', palette='Reds_d', dodge=False, ax=ax, legend=False)
for index, row in df_h3.iterrows():
    ax.text(row.name, row['U-Statistik'] + 10, f'p={row["p-Wert"]:.3e}', color='black', ha="center")  # Adjust the y position
ax.set_title('Mann-Whitney-U-Testergebnisse für Hypothese 3')
##########################################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv('Data.csv')
data['Q3_Familienbesitz'] = data['Q3_Familienbesitz'].astype('category')
# Convert all other columns to numeric
for col in data.columns[3:]:
    data[col] = pd.to_numeric(data[col], errors='coerce')
# Separate data for family and non-family businesses
data_family = data[data['Q3_Familienbesitz'] == 1]
data_non_family = data[data['Q3_Familienbesitz'] == 2]
# Calculate the correlation matrices
corr_family = data_family.iloc[:, 3:].corr()
corr_non_family = data_non_family.iloc[:, 3:].corr()
# Plot heatmaps
fig, axs = plt.subplots(1, 2, figsize=(20, 10))
# Family businesses heatmap
sns.heatmap(corr_family, annot=True, fmt=".2f", cmap='coolwarm', center=0, ax=axs[0])
axs[0].set_title('Korrelationsmatrix für Familienunternehmen')
axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=45, horizontalalignment='right')
axs[0].set_yticklabels(axs[0].get_yticklabels(), rotation=0)
# Non-family businesses heatmap
sns.heatmap(corr_non_family, annot=True, fmt=".2f", cmap='coolwarm', center=0, ax=axs[1])
axs[1].set_title('Korrelationsmatrix für Nicht-Familienunternehmen')
axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=45, horizontalalignment='right')
axs[1].set_yticklabels([])  # Remove y-axis labels
# Adjust layout
plt.tight_layout()
# Save the figure
plt.savefig('heatmap_comparison.png')
plt.show()
