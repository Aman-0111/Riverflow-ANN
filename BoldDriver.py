import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import copy

"""

Note that the program is largely the same as the main Neural network program with changes made to implement bold driver.
All new changes are commented by three hashtags.
ETC:

### Finding difference in weights
x = 12-y

"""


np.random.seed(0) 

# Class for handelling excel values
class Data:
    def __init__(self,filename,sheet):
        """Constructor for my data class

        Args:
            filename (String): Name of excel file
            sheet (String): Name of excel sheet
        """
        self.df = pd.read_excel (filename+'.xlsx', sheet_name= sheet)

    def getColumns(self):
        """Returns Column titles

        Returns:
            String list: Column titles
        """
        return list(self.df.columns.values.tolist())

    def getNumColumn(self):
        """Returns number of columns in excel sheet

        Returns:
            Int: Number of columns
        """
        columns = list(self.getColumns())
        return len(columns[1:-1])
    
    def getNumRows(self):
        """Returns number of rows in excel sheet

        Returns:
            Int: Number of rows
        """
        l = self.getColumns()
        dates = self.df[l[0]]
        return len(dates)

    def getTrainValues(self):
        """Returns predictor values in excel sheet

        Returns:
            2D list: values in each predictor column
        """
        l = self.getColumns()[1:-1]
        values = []
        for elm in (l):
            val = list(self.df[elm])
            values.append(val)
        return(values)

    def getExpectedValues(self):
        """Returns all values for predictand

        Returns:
            list: All predictand values        
        """
        l = self.getColumns()[-1]
        values = list(self.df[l])
        return(values)


    def plotGraph(self,epochs,rmse):
        """Creates Graph of RMSE against Epochs

        Args:
            epochs (Int List): List of 1 to number of epochs run
            rmse (Float): List of RMSE's
        """

        plt.plot(epochs,rmse)
        plt.ylabel("RMSE")
        plt.xlabel("Epochs")
        plt.legend()
        plt.show()

# Class for standardising values
class Standard:
    def __init__(self,data):
        """Constructor for Standard class Sets a data attribute

        Args:
            data (Data): Data class shown above
        """
        self.data = data

    def maxMin (self):
        """Obtains maximum and minimum values in data
        """

        values = self.data.getTrainValues()

        # Gets only the training and validation values 
        for x in range(len(values)):
            values[x] = values[x][:1095]

        self.mins = []
        self.maxs = []
        # Creates list of max and min values in each column
        for elm in values:
            self.mins.append(np.amin(elm))
            self.maxs.append(np.amax(elm))   

    def standInp(self):
        """Standardises the training data

        Returns:
            2D list: Standardised values for each column
        """

        values = self.data.getTrainValues()

        # Gets only the training values
        for x in range(len(values)):
            values[x] = values[x][:732]

        table = []
        # Applies the standardisation formula to each value and appends it to a list
        for y in range (len(values)):
            column = []
            for x in range (len(values[0])):
                column.append(self.standFormula(values[y][x],self.mins[y],self.maxs[y]))
            table.append(column)
        return table

    def standValidation(self):
        """Standardises validation values

        Returns:
            2d list: Standardised validation values
        """
        values = self.data.getTrainValues()
        
        # Gets only the validation values
        for x in range(len(values)):
            values[x] = values[x][733:1095]

        table = []
        # Applies the standardisation formula to each value and appends it to a list
        for y in range (len(values)):
            column = []
            for x in range (len(values[0])):
                column.append(self.standFormula(values[y][x],self.mins[y],self.maxs[y]))
            table.append(column)

        return table

    def actualTrainVals(self):
        """Standardises predictand values for training set
        """
        self.fin = self.data.getExpectedValues()[:1095]

        # Adds max and min values to the list
        self.mins.append(np.amin(self.fin))
        self.maxs.append(np.amax(self.fin))

        self.fin = self.data.getExpectedValues()[:732]

        # Standardises and appends training predictand set
        column = []
        for x in range (len(self.fin)):
            column.append(self.standFormula(self.fin[x],self.mins[-1],self.maxs[-1]))

        self.fin = column

    def actualVal(self):
        """Standardises predictand values for validation data set

        Returns:
            list: Standardised values of predictand
        """
        predictand = self.data.getExpectedValues()[733:1095]

        column = []
        # Standardises and appends validation predictand set
        for x in range (len(predictand)):
            column.append(self.standFormula(predictand[x],self.mins[-1],self.maxs[-1]))

        return column    
    
    def allExpected(self):
        """Obtains all predictand values

        Returns:
            list: list of all standardised predictand values
        """
        predictand = self.data.getExpectedValues()

        column = []
        # Standardises and appends the entire predictand set
        for x in range (len(predictand)):
            column.append(self.standFormula(predictand[x],self.mins[-1],self.maxs[-1]))

        return column  

    def allPredict(self):
        """Obtains all predictor values

        Returns:
            2D list: contains all predictor values
        """
        values = self.data.getTrainValues()

        table = []
        # Applies the standardisation formula to each value and appends it to a list
        for y in range (len(values)):
            column = []
            for x in range (len(values[0])):
                column.append(self.standFormula(values[y][x],self.mins[y],self.maxs[y]))
            table.append(column)

        return table

    @staticmethod
    def standFormula(val,min,max):
        """Standardises value

        Args:
            val (Float): value to be standardised
            min (Float): min value for that column
            max (Float): max value for that column

        Returns:
            Float: standardised value
        """
        return 0.8*((val - min)/(max-min))+0.1
    
    @staticmethod
    def deStandFormula(val,min,max):
        """Destandardises value

        Args:
            val (Float): value to be destandardised
            min (Float): min value for that column
            max (Float): max value for that column

        Returns:
            Float: destandardised value
        """
        return (((val-0.1)/0.8) * (max-min)) + min
        


# Class handles everything neural network related
class Mlp:
    def __init__(self, network_layout = (3,7,1)):
        """Constructor for mlp class

        Args:
            network_layout (tuple, optional): Hold number of nodes in each layer. Defaults to (3,5,1).
        """
        # Learning rate, sum of values during training phase (for RMSE calcs), sum of values during validation (for RMSE calcs) and stores RMSE values
        self.lrate = 0.01
        self.train_sum = 0
        self.valid_sum = 0
        self.rmse = []
        # Holds layout of weight matrices, holds all biases and holds all weights in neural network
        self.weight_layout = []
        self.biases = []
        self.weights = []
        ### Holds previous weights and biases
        self.prev_weights = []
        self.prev_biases = []

        # Appends shape of weight matrices to list
        for elm in zip(network_layout[1:],network_layout[:-1]):
            self.weight_layout.append(elm)
        
        # Creates random weight matrices for the network
        for layout in self.weight_layout:
            self.weights.append(np.random.uniform(-(2/layout[-1]),(2/layout[-1]),layout))
        
        # Creates list of zeros for initial biases
        for size in network_layout[1:]:
            self.biases.append(np.zeros(size))

    def validProp(self,val,actual):
        """Forward propogation when using validation data

        Args:
            val (Float List): Row of values in excel
            actual (Float): Value of predictand in row
        """
        # Calculates activation values for each node then changes val to be used in next layer
        for weight,bias in zip(self.weights,self.biases):
            multi_res = np.matmul(weight,val)
            multi_res += bias
            val = self.sigmoidFunc(multi_res)
        
        # Calculates the square of the difference for RMSE
        self.valid_sum += (val - actual)**2
       
    def forwardProp(self,val):
        """Forward propogation when using training data

        Args:
            val (Float List): Row of values in excel
        """
        # Stores the values at each layer
        self.layers_vals = [val]

        # Same as loop in method above except each value gets added to the list
        for weight,bias in zip(self.weights,self.biases):
            multi_res = np.matmul(weight,val)
            multi_res += bias
            val = self.sigmoidFunc(multi_res)
            self.layers_vals.append(list(val))
        
    def backProp(self,actual):
        """Back propogation when using training data

        Args:
            actual (Float): Value of predictand in row
        """

        ### Storse deep copies of biases and weights
        self.prev_weights = copy.deepcopy(self.weights)
        self.prev_biases = copy.deepcopy(self.biases)

        # Stores delta of output node in list
        self.delts = [[(actual - self.layers_vals[-1][0])*(self.dervSigmoid(self.layers_vals[-1][0]))]]

        # Calculates delta for each layer and appends to list
        layer = []
        for x in range (len(self.weights[-1][0])):
            layer.append(self.weights[-1][0][x] * self.delts[0][0] * self.dervSigmoid(self.layers_vals[-2][x]))
        self.delts.append(layer)
        
        # Next 3 loops updates all weights in the network
        for x in range (len(self.weights[-1][0])):
            self.weights[-1][0][x] = self.weights[-1][0][x] + self.lrate * self.delts[0][0] * self.layers_vals[1][x]

        for y in range (len(self.weights[0])):
            for z in range (len(self.weights[0][y])):
                self.weights[0][y][z] = self.weights[0][y][z] + self.lrate * self.delts[1][y] * self.layers_vals[0][z]

        # Updates biases using delta values
        for n in range (len(self.biases)):
            for m in range (len(self.biases[n])):
                self.biases[n][m] -= self.lrate * self.delts[-(n+1)][m]

        # Calculates the square of the difference for RMSE
        self.train_sum += (self.layers_vals[-1][0] - actual)**2

    def calcRmse(self,size):
        """Calculates RMSE of epoch for training

        Args:
            size (int): Number of rows in the training data set
        """
        rmse = math.sqrt(self.train_sum/size)
        self.rmse.append(rmse)

    def valRmse(self,size):
        """Calculates RMSE of epoch for validation

        Args:
            size (Int): Number of rows in the training data set

        Returns:
            Float: RMSE value
        """
        rmse = math.sqrt(self.valid_sum/size)
        return rmse

    def setTotal(self):
        """Resets values for next epoch
        """
        self.valid_sum = 0
        self.train_sum = 0

    def getRmse(self):
        """Returns RMSE of training data
        """
        return self.rmse

    def sigmoidFunc(self,val):
        """Calculates value of sigmoid function

        Args:
            val (float): result at node

        Returns:
            Float: Sigmoid value
        """
        return 1/(1+np.exp(-val))

    def dervSigmoid(self,val):
        """Calculates derivative of sigmoid function

        Args:
            val (float): activation value at node

        Returns:
            Float: derivative of activation value
        """
        return val*(1-val)

    ### The remaining methods were added to support bold driver
    def getMse(self,size):
        """Returns mse

        Returns:
            Float: MSE after 1 epoch
        """
        return self.train_sum/size

    def setLrate(self,change):
        """Sets learning rate based on MSE

        Args:
            change (Char): holds weather lrate should increase or decrease
        """
        if change == "d":
            self.lrate *= 0.7
        else:
            self.lrate *= 1.05

    def resetWeights(self):
        """Returns weights to previous state
        """
        self.weights = copy.deepcopy(self.prev_weights)
        self.biases = copy.deepcopy(self.prev_biases)


if __name__ == "__main__":

    # User to enters details
    filename = input("Enter File name/Directory (only if located in a diffrent location):")
    sheetname = input("Enter Sheet name:")

    iNodes = int(input("Enter Number of input nodes:"))
    hNodes = int(input("Enter Number of hidden nodes:"))
    oNodes = int(input("Enter Number of output nodes:"))

    # Instantiating all the classes
    dataset = Data(filename,sheetname)
    stan = Standard(dataset)
    ann = Mlp((iNodes,hNodes,oNodes))
    
    # Sets all standardised values
    stan.maxMin()
    stan.actualTrainVals()
    indata = stan.standInp()
    validate = stan.standValidation()

    ### List of MSE for implementing bold driver 
    MSE = []

    # Used in validation
    checker = []

    # Epoch loop
    for x in range (10000):
        ann.setTotal()
        # Loops through all training values
        for y in range (len(indata[0])):
            
            # Gets the same row value from each column and puts it in a list
            inp = [i[y] for i in indata]

            ann.forwardProp(inp)
            ann.backProp(stan.fin[y])
            
        ### Runs bold driver every 25 epochs
        if (not(x%25)):
            MSE.append(ann.getMse(len(indata[0]))*100)
            if (x>=1):
                change = MSE[-1] - MSE[-2]
                ### Increases or Decreases learning rate based on how big of a change was made
                if (change > -1)  and (change < 0):
                    ann.setLrate("i")
                elif (change > 1):
                    ann.setLrate("d")
                    ann.resetWeights()          
                       
        # Runs validation every 25 epochs
        if not(x%25):
            
            # Runs forward propogation for all rows in validation set
            for z in range (len(validate[0])):
                val = [v[z] for v in validate]
                ann.validProp(val,stan.actualVal()[z])

            # Appends RMSE values to list to check for over fitting
            checker.append(ann.valRmse(361))
            if (len(checker) != 1 and len(checker) != 0):
                # Breaks loop when signs of over fitting are found
                if checker[-1] > checker[-2]:
                    break
    
        ann.calcRmse(len(indata[0]))
            
    dataset.plotGraph(list(range(1,x+1)),ann.getRmse())


    """
    Used to plot expected vs actual
    """

    # modelled = []
    # all_inputs = stan.allPredict()
    # for m in range (len(all_inputs[0])):
    #     inp = [n[m] for n in all_inputs]
    #     modelled.append(ann.validProp(inp,3))

    # plt.plot(dataset.getDates(),stan.allExpected() , label = "Expected")
    # plt.plot(dataset.getDates(),modelled , label = "Modelled")
    # plt.ylabel("Standardise Skelton")
    # plt.xlabel("Date")
    # plt.legend()
    # plt.show()