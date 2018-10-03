import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

data = pd.read_fwf('brain_body.txt')
brain = data[['Brain']]
body = data[['Body']]

# Create linear regression object
regr = linear_model.LinearRegression()

regr.fit(brain.head(3), body.head(3))

predicted_data = regr.predict(pd.DataFrame({"values":[3.385]}))



challenge_dataframe = pd.read_table('challenge_dataset.txt', delim_whitespace=False, names=("testing", ))
challenge_dataframe_xvalues = pd.DataFrame(challenge_dataframe.testing.str.split(",").tolist(),
                                           columns=["xvalue", "yvalue"])[[0]]

print(challenge_dataframe_xvalues.head())

challenge_dataframe_yvalues = pd.DataFrame(challenge_dataframe.testing.str.split(",").tolist(),
                                           columns=["xvalue", "yvalue"])[[1]]
print(predicted_data.shape)
print(predicted_data)
print(challenge_dataframe_yvalues.shape)
print(challenge_dataframe_yvalues.head())


# Plot outputs
plt.scatter(challenge_dataframe_xvalues, challenge_dataframe_yvalues,  color='black')
plt.plot(challenge_dataframe_xvalues, predicted_data, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()