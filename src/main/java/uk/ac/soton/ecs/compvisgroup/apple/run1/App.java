package uk.ac.soton.ecs.compvisgroup.apple.run1;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.BasicClassificationResult;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

/**
 * OpenIMAJ Hello world!
 * @author Oscar van Leusen
 */
public class App {
    public static void main(String[] args) throws FileSystemException {
        TinyImageProcessor ti = new TinyImageProcessor(24);
        KNearest knn = new KNearest(20);
        boolean useValidation = false; // Create an additional validation dataset from the labelled training data. Should be false for final model.


        System.out.println("Loading Train and Test datasets. Ensure training.zip and testing.zip are inside the resources folder.");
        String resourcesDir = App.class.getClassLoader().getResource("").toString();
        VFSGroupDataset<FImage> labelledData = new VFSGroupDataset<>("zip:" + resourcesDir + "training.zip", ImageUtilities.FIMAGE_READER);
        VFSListDataset<FImage> testing = new VFSListDataset<>("zip:" + resourcesDir + "testing.zip", ImageUtilities.FIMAGE_READER);

        List<double[]> tinyImageVectors = new ArrayList<>();
        List<String> classes = new ArrayList<>();

        if (useValidation) {
            //Split the Labelled Train set for validation purposes.
            GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter<>(labelledData, 80, 0, 20);
            GroupedDataset<String, ListDataset<FImage>, FImage> training = splits.getTrainingDataset();
            GroupedDataset<String, ListDataset<FImage>, FImage> validation = splits.getTestDataset();

            // Train on split dataset
            for (String imageClass : training.getGroups()) {
                for (FImage image : training.get(imageClass)) {
                    tinyImageVectors.add(ti.extractFeature(image).values);
                    //System.out.println(imageClass);
                    classes.add(imageClass);
                }
            }

            // Train KNN Classifier
            knn.train(tinyImageVectors, classes);

            // Validate accuracy of KNN Classifier (In my tests, about 14.2% accuracy with 80/20 split)
            knn.validate(ti, validation.entrySet());
        } else {
            // For some reason VFSGroupDataset includes every image twice with the 'training' label as well as the
            // expected class labels. So discard anything labelled as 'training' (otherwise it predicts everything as 'training').
            labelledData.remove("training");

            // Train on full dataset
            for (String imageClass : labelledData.getGroups()) {
                for (FImage image : labelledData.get(imageClass)) {
                    tinyImageVectors.add(ti.extractFeature(image).values);
                    //System.out.println(imageClass);
                    classes.add(imageClass);
                }
            }

            // Train KNN Classifier
            knn.train(tinyImageVectors, classes);
        }


        //Classify unlabelled data
        Map<String, String> predictions = new TreeMap<>(new FileNameComparator());
        for (int i=0; i<testing.size()-1; i++) {
            String fileName = testing.getFileObject(i).getName().getBaseName();
            BasicClassificationResult<String> result = knn.classify(ti, testing.get(i));
            String predictedClass = result.getPredictedClasses().iterator().next();
            predictions.put(fileName, predictedClass);
        }

        // Write predictions to file
        try {
            FileWriter writer = new FileWriter("run1.txt");
            for (Map.Entry<String, String> prediction : predictions.entrySet()) {
                writer.write(prediction.getKey() + " " + prediction.getValue() + System.lineSeparator());
            }
            writer.close();
            System.out.println("Results written to run1.txt");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Custom comparator for sorting Test image filenames by their number only.
     */
    static class FileNameComparator implements Comparator<String> {

        @Override
        public int compare(String o1, String o2) {
            String notNumberRegex = "[^\\d]";
            int file1 = Integer.parseInt(o1.replaceAll(notNumberRegex, ""));
            int file2 =  Integer.parseInt(o2.replaceAll(notNumberRegex, ""));
            return file1 - file2;
        }
    }


}
