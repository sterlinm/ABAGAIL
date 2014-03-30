package shared.tester;

import shared.Instance;
import shared.reader.DataSetLabelBinarySeperator;
import util.linalg.Vector;

/**
 * A test metric for accuracy.  This metric reports of % correct and % incorrect for a test run.
 * 
 * @author Jesse Rosalia <https://github.com/theJenix>
 * @date 2013-03-05
 */
public class AccuracyTestMetric implements TestMetric {

    private int count;    
    private int countCorrect;
    private double epsilon = 1e-6;
    
    @Override
    public void addResult(Instance expected, Instance actual) {
        Comparison c = new Comparison(expected, actual);
        c.setEpsilon(this.epsilon);

        count++;
        if (c.isAllCorrect()) {
            countCorrect++;
        }
    }
    
    public double getResult() {
        return count > 0 ? ((double)countCorrect)/count : 1; //if count is 0, we consider it all correct
    }

    public void printResults() {
        //only report results if there were any results to report.
        if (count > 0) {
            double pctCorrect   = getResult();
            double pctIncorrect = (1 - pctCorrect);
            System.out.println(String.format("Correctly Classified Instances: %.02f%%",   100 * pctCorrect));
            System.out.println(String.format("Incorrectly Classified Instances: %.02f%%", 100 * pctIncorrect));
        } else {

            System.out.println("No results added.");
        }

    }

    public void setEpsilon(double e) {
        this.epsilon = e;
    }
}
