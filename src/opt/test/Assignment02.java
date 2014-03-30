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
	// Get SPAM Data
    private static DataSet[] spamDatasets = getSpamData();
    private static DataSet spamTrain = spamDatasets[0];
    private static DataSet spamTest = spamDatasets[1];

    // General Neural Network Settings
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
    private static ErrorMeasure measure = new SumOfSquaresError();
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];
    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3];
    private static String[] oaNames = {"BackProp", "RHC", "SA", "GA"};
    private static DecimalFormat df = new DecimalFormat("0.000");

    // SPAM Neural Network Settings
    private static BackPropagationNetwork spamNetworks[] = new BackPropagationNetwork[4];
    private static NeuralNetworkTester[] spamNetTesters = new NeuralNetworkTester[4];
    private static int spamInputLayer = 57, spamHiddenLayer = 14, spamOutputLayer = 1, spamTrainingIterations = 1000;  

    /**
     * The test main
     * @param args ignored parameters
     */
    public static void main(String[] args) throws Exception {

        // Spam Network - Backpropagation
        spamNetworks[0] = factory.createClassificationNetwork(
           new int[] {spamInputLayer, spamHiddenLayer, spamOutputLayer});
        ConvergenceTrainer trainer = new ConvergenceTrainer(
            new BatchBackPropagationTrainer(spamTrain, spamNetworks[0],
                new SumOfSquaresError(), new RPROPUpdateRule()),1E-10,spamTrainingIterations);
        trainer.train();

        // Spambase Network - Optimization Algorithms
        for(int i = 0; i < oa.length; i++) {
            spamNetworks[i+1] = factory.createClassificationNetwork(
                new int[] {spamInputLayer, spamHiddenLayer, spamOutputLayer});
            nnop[i] = new NeuralNetworkOptimizationProblem(spamTrain, spamNetworks[i+1], measure);
        }

        oa[0] = new RandomizedHillClimbing(nnop[0]);
        oa[1] = new SimulatedAnnealing(1E11, .95, nnop[1]);
        oa[2] = new StandardGeneticAlgorithm(200, 100, 10, nnop[2]);

        for(int i = 0; i < oa.length; i++) {
            train(oa[i], spamNetworks[i+1], oaNames[i+1], 1000, spamTrain.getInstances(), spamTest.getInstances(),true);

            Instance optimalInstance = oa[i].getOptimal();
            spamNetworks[i+1].setWeights(optimalInstance.getData());

        }

        for(int i = 0; i < spamNetworks.length; i++) {
            TestMetric trainResults = printNNetResults(spamNetworks[i], spamTrain.getInstances());
            System.out.println("\nResults for " + oaNames[i] + " (Training):");
            trainResults.printResults();

            TestMetric testResults = printNNetResults(spamNetworks[i], spamTest.getInstances());
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
        System.out.println("\nError results for " + oaName + "\n---------------------------");

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

        DataSet[] ds = new DataSet[2];

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

        DataSet spamDS = new DataSet(instances);
        TestTrainSplitFilter ttf = new TestTrainSplitFilter(30);
        ttf.filter(spamDS);
        DataSet spamTrain = ttf.getTrainingSet();
        DataSet spamTest = ttf.getTestingSet();

        ds[0] = spamTrain;
        ds[1] = spamTest;

        return ds;
    }
}