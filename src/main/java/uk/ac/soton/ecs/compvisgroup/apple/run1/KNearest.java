package uk.ac.soton.ecs.compvisgroup.apple.run1;

import org.openimaj.data.dataset.ListDataset;
import org.openimaj.experiment.evaluation.classification.BasicClassificationResult;
import org.openimaj.feature.DoubleFV;
import org.openimaj.image.FImage;
import org.openimaj.knn.DoubleNearestNeighboursExact;
import org.openimaj.util.pair.IntDoublePair;
import java.util.*;
import java.util.stream.Collectors;

/**
 * @author Oscar van Leusen
 */
class KNearest {

    // DoubleNearestNeighboursExact requires feature vectors as double[][], not a List of DoubleFV.
    private DoubleNearestNeighboursExact knn;
    private int k;
    private List<String> classes;

    KNearest(int k) {
        this.k = k;
        }


    void train(List<double[]> trainVectors, List<String> classes) {
        this.knn = new DoubleNearestNeighboursExact(trainVectors.toArray(new double[][]{}));
        this.classes = classes;
    }

    public BasicClassificationResult<String> classify(TinyImageProcessor featureExtractor, FImage image) {
        DoubleFV tinyImageVector = featureExtractor.extractFeature(image);

        // Find neighbours in KNN
        List<IntDoublePair> result = knn.searchKNN(tinyImageVector.values, this.k);


        Map<String, Integer> neighbourClassCount = new HashMap<>();
        // For each neighbour found, identify what class that neighbour is in and create a count.
        for (IntDoublePair neighbour : result) {
            // Of the IntDoublePair, the Int corresponds with the index of the trainVector and classes label list, so we
            // can get the class label for that neighbour.
            String imageClass = "null";
            // If k-th nearest neighbour cannot be determined, returns index of -1. Without this condition I experienced
            // IndexOutOfBoundsExceptions.
            if (neighbour.first != -1) {
                imageClass = classes.get(neighbour.first);
            }

            // Increment or create the count for that label.
            if (neighbourClassCount.containsKey(imageClass)) {
                int count = neighbourClassCount.get(imageClass) + 1;
                neighbourClassCount.put(imageClass, count);
            } else {
                neighbourClassCount.put(imageClass, 1);
            }
        }

        // Sort the neighbourClassCount descending to get most frequent neighbour classes.
        LinkedHashMap<String, Integer> sortedClassCount = neighbourClassCount.entrySet().stream()
                .sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
                .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue,
                        (e1, e2) -> e1, LinkedHashMap::new));

        // Get the top prediction
        String predictedClass = sortedClassCount.keySet().iterator().next();
        int count = sortedClassCount.get(predictedClass);

        // Calculate the confidence
        double confidence = count / (double) this.k;
        BasicClassificationResult<String> classificationResult = new BasicClassificationResult<>();
        classificationResult.put(predictedClass, confidence);
        return classificationResult;
    }

    public void validate(TinyImageProcessor featureExtractor, Set<Map.Entry<String, ListDataset<FImage>>> validationDataset) {
        double correct = 0;
        double incorrect = 0;

        for (Map.Entry<String, ListDataset<FImage>> images : validationDataset) {
            System.out.println("Evaluating key : " + images.getKey());
            for (FImage image : images.getValue()) {
                BasicClassificationResult<String> result = this.classify(featureExtractor, image);

                // Increment counts according to whether prediction is the same as the label
                if (result.getPredictedClasses().iterator().next().equals(images.getKey())) {
                    correct += 1;
                } else {
                    incorrect += 1;
                }
            }
        }

        System.out.println("Accuracy: " + (correct / (correct + incorrect)));
    }

}
