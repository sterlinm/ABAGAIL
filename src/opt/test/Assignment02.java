package opt.test;

import dist.*;
import opt.*;
import opt.example.*;
import opt.ga.*;
import shared.*;
import func.nn.backprop.*;
import shared.reader.*;
import shared.filt.*;
import shared.tester.*;

import java.util.*;
import java.io.*;
import java.text.*;

/**
 * Assignment 2
 *
 * @author Michael Sterling
 * @version 1.0
 */

public class Assignment02 {
	// Get Data
    private static DataSet[] spamDatasets = getSpamData();
    private static DataSet[] glassDatasets = getGlassData();
    private static DataSet trainDS;
    private static DataSet valDS;
    private static DataSet testDS;

    // General Neural Network Settings
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
    private static ErrorMeasure measure = new SumOfSquaresError();
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];
    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3];
    private static String[] oaNames = {"BackProp", "RHC", "SA", "GA"};
    private static DecimalFormat df = new DecimalFormat("0.000");
    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[4];
    private static int inputLayer, hiddenLayer, outputLayer, numIters = 1000;  

    /**
     * The test main
     * @param args ignored parameters
     */
    public static void main(String[] args) throws Exception {
        //Spam Data
        trainDS = spamDatasets[0];
        valDS = spamDatasets[1];
        testDS = spamDatasets[2];
        inputLayer = 57;
        hiddenLayer = 14;
        outputLayer = 1;

        // Glass Data
        // trainDS = glassDatasets[0];
        // valDS = glassDatasets[1];
        // testDS = glassDatasets[2];
        // inputLayer = 9;
        // hiddenLayer = 10;
        // outputLayer = 1;

        // Experiment with number of nodes
        int maxNodes = 20;
        double nodeTestPerformance[] = new double[maxNodes];
        System.out.println("\nVarying the number of nodes in the hidden layer...\n---------------------------");
        for (int i = 1; i <= maxNodes; i++) {
            networks[0] = factory.createClassificationNetwork(
                new int[] {inputLayer, i, outputLayer});
            ConvergenceTrainer trainer = new ConvergenceTrainer(
                new BatchBackPropagationTrainer(trainDS, networks[0],
                    new SumOfSquaresError(), new RPROPUpdateRule()),1E-10,numIters);
            trainer.train();
            TestMetric nodeTrainResults = printNNetResults(networks[0], trainDS.getInstances());
            TestMetric nodeValResults = printNNetResults(networks[0], valDS.getInstances());
            System.out.println(i + "\t" + df.format(nodeTrainResults.getResult()) + "\t" + df.format(nodeValResults.getResult()));
        }

        // Backpropagation
        networks[0] = factory.createClassificationNetwork(
           new int[] {inputLayer, hiddenLayer, outputLayer});
        ConvergenceTrainer trainer = new ConvergenceTrainer(
            new BatchBackPropagationTrainer(trainDS, networks[0],
                new SumOfSquaresError(), new RPROPUpdateRule()),1E-10,numIters);
        trainer.train();

        // Optimization Algorithms
        for(int i = 0; i < oa.length; i++) {
            networks[i+1] = factory.createClassificationNetwork(
                new int[] {inputLayer, hiddenLayer, outputLayer});
            nnop[i] = new NeuralNetworkOptimizationProblem(trainDS, networks[i+1], measure);
        }

        oa[0] = new RandomizedHillClimbing(nnop[0]);
        oa[1] = new SimulatedAnnealing(1E11, .95, nnop[1]);
        oa[2] = new StandardGeneticAlgorithm(200, 100, 10, nnop[2]);

        for(int i = 0; i < oa.length; i++) {
            train(oa[i], networks[i+1], oaNames[i+1], 1000, trainDS.getInstances(), testDS.getInstances(),false);

            Instance optimalInstance = oa[i].getOptimal();
            networks[i+1].setWeights(optimalInstance.getData());

        }

        for(int i = 0; i < networks.length; i++) {
            TestMetric trainResults = printNNetResults(networks[i], trainDS.getInstances());
            System.out.println("\nResults for " + oaNames[i] + " (Training):");
            trainResults.printResults();

            TestMetric testResults = printNNetResults(networks[i], testDS.getInstances());
            System.out.println("\nResults for " + oaNames[i] + " (Test):");
            testResults.printResults();
        }
    }

    private static TestMetric printNNetResults(BackPropagationNetwork net, Instance[] instances) {
        AccuracyTestMetric testMetric = new AccuracyTestMetric();
        testMetric.setEpsilon(0.5);
        NeuralNetworkTester netTester = new NeuralNetworkTester(net,testMetric);
        netTester.test(instances);
        return netTester.getMetric(0);
    }

    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName, double numIters, Instance[] trainInstances, Instance[] testInstances, Boolean printFlag) {
        if (printFlag) {
            System.out.println("\nError results for " + oaName + "\n---------------------------");
        }

        for(int i = 0; i < numIters; i++) {
            oa.train();

            double error = 0;
            for(int j = 0; j < trainInstances.length; j++) {
                network.setInputValues(trainInstances[j].getData());
                network.run();

                Instance output = trainInstances[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                error += measure.value(output, example);
            }

            network.setWeights(oa.getOptimal().getData());
            TestMetric trainResults = printNNetResults(network,trainInstances);
            TestMetric testResults = printNNetResults(network,testInstances);
            if (printFlag) {
                System.out.println(df.format(error) + "\t" + df.format(trainResults.getResult()) + "\t" + df.format(testResults.getResult()));
            }
        }
    }

    private static DataSet[] getSpamData() {

        DataSet[] ds = new DataSet[3];
        DataSet tempDS;

        double[][][] attributes = new double[4601][][];

        try {
            // BufferedReader br = new BufferedReader(new FileReader(new File("src/opt/test/abalone.txt")));
            BufferedReader br = new BufferedReader(new FileReader(new File("src/opt/test/spambase_randomized.data")));

            for(int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[57]; // 57 attributes
                attributes[i][1] = new double[1];

                for(int j = 0; j < 57; j++)
                    attributes[i][0][j] = Double.parseDouble(scan.next());

                attributes[i][1][0] = Double.parseDouble(scan.next());
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        Instance[] instances = new Instance[attributes.length];

        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            instances[i].setLabel(new Instance(attributes[i][1][0]));
        }

        DataSet allDS = new DataSet(instances);
        TestTrainSplitFilter ttf = new TestTrainSplitFilter(40);
        ttf.filter(allDS);
        ds[0] = ttf.getTrainingSet(); // training set
        tempDS = ttf.getTestingSet();
        ttf = new TestTrainSplitFilter(50);
        ttf.filter(tempDS);
        ds[1] = ttf.getTrainingSet(); // validation set
        ds[2] = ttf.getTestingSet(); // test set

        return ds;
    }

    private static DataSet[] getGlassData() {

        DataSet[] ds = new DataSet[3];
        DataSet tempDS;

        double[][][] attributes = new double[214][][];

        try {
            BufferedReader br = new BufferedReader(new FileReader(new File("src/opt/test/glassData.csv")));

            for(int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[9]; // 9 attributes
                attributes[i][1] = new double[1];

                for(int j = 0; j < 9; j++)
                    attributes[i][0][j] = Double.parseDouble(scan.next());

                attributes[i][1][0] = Double.parseDouble(scan.next());
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        Instance[] instances = new Instance[attributes.length];

        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            instances[i].setLabel(new Instance(attributes[i][1][0]));
        }

        DataSet allDS = new DataSet(instances);
        TestTrainSplitFilter ttf = new TestTrainSplitFilter(40);
        ttf.filter(allDS);
        ds[0] = ttf.getTrainingSet(); // training set
        tempDS = ttf.getTestingSet();
        ttf = new TestTrainSplitFilter(50);
        ttf.filter(tempDS);
        ds[1] = ttf.getTrainingSet(); // validation set
        ds[2] = ttf.getTestingSet(); // test set

        return ds;
    }
}