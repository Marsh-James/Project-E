package uk.ac.soton.lw2n17;

import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import de.bwaldvogel.liblinear.SolverType;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.feature.local.list.MemoryLocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.keypoints.FloatKeypoint;
import org.openimaj.ml.annotation.ScoredAnnotation;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.util.pair.IntFloatPair;

public class LinearClassifier {
    public static void main( String[] args ) throws Exception {
        // Training dataset
        VFSGroupDataset<FImage> raw = new VFSGroupDataset<>(
                "YOUR_PATH_HERE", ImageUtilities.FIMAGE_READER
        );

        // Used for splitting training data into 80-20 train test split
        GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter<>(
                raw, 80, 0, 20
        );

        // Used for training quantiser later
        ArrayList<FImage> training = new ArrayList<>();

        for (Map.Entry<String, VFSListDataset<FImage>> entry : raw.entrySet()) {
            training.addAll(entry.getValue());
        }
        System.out.println("Got the data");

        long start = System.currentTimeMillis();
        HardAssigner<float[], float[], IntFloatPair> assigner = trainQuantiser(training, 50000);
        long end = System.currentTimeMillis();
        System.out.println("Finished getting vocab, took " + ((end - start) / 1000) + "s");

        start = System.currentTimeMillis();
        FeatureExtractor<SparseIntFV, FImage> extractor = new MyExtractor(
                assigner, new MyEngine(4, 4)
        );

        LiblinearAnnotator<FImage, String> ann = new LiblinearAnnotator<>(
                extractor, LiblinearAnnotator.Mode.MULTICLASS,
                SolverType.L1R_L2LOSS_SVC, 1.0, 0.00001
        );

        ann.train(raw);
        end = System.currentTimeMillis();

        System.out.println("Finished training, took " + ((end - start) / 1000) + "s");

        // Eval
        /*
        ClassificationEvaluator<CMResult<String>, String, FImage> eval = new ClassificationEvaluator<>(
                ann, splits.getTestDataset(), new CMAnalyser<FImage, String>(CMAnalyser.Strategy.SINGLE)
        );

        Map<FImage, ClassificationResult<String>> guesses = eval.evaluate();
        for (ClassificationResult<String> entry : guesses.values()) {
            System.out.println(entry);
        }

        CMResult<String> result = eval.analyse(guesses);

        System.out.println(result.getDetailReport());
        */

        // Generate the output test file
        VFSListDataset<FImage> test = new VFSListDataset<>(
                "YOUR_PATH_HERE", ImageUtilities.FIMAGE_READER
        );
        int count = 0;
        FileWriter writer = new FileWriter("run2.txt");

        for (FImage pic : test) {
            List<ScoredAnnotation<String>> result = ann.annotate(pic);
            writer.write(count + ".jpg " + result.get(0).annotation + System.lineSeparator());
            count++;
        }

        writer.close();
    }

    static HardAssigner<float[], float[], IntFloatPair> trainQuantiser(
            ArrayList<FImage> sample,
            int sampleSize
    ) {
        MemoryLocalFeatureList<FloatKeypoint> allFeatures = new MemoryLocalFeatureList<>();
        MyEngine myEngine = new MyEngine(4, 4);

        for (FImage image : sample) {
            allFeatures.addAll(myEngine.analyze(image));
        }

        // To ensure the list is randomly sampled
        Collections.shuffle(allFeatures);

        if (allFeatures.size() > sampleSize) {
            allFeatures = allFeatures.subList(0, sampleSize);
        }

        DataSource<float[]> dataSource = new LocalFeatureListDataSource<>(allFeatures);

        // Establishing the vocab
        FloatKMeans km = FloatKMeans.createKDTreeEnsemble(500);
        FloatCentroidsResult result = km.cluster(dataSource);

        return result.defaultHardAssigner();
    }

    static class MyEngine {
        int windowSize;
        int stepSize;

        MyEngine(int windowSize, int stepSize) {
            this.windowSize = windowSize;
            this.stepSize = stepSize;
        }

        LocalFeatureList<FloatKeypoint> analyze(FImage image) {
            MemoryLocalFeatureList<FloatKeypoint> output = new MemoryLocalFeatureList<>();

            // Moving the window across the image
            for (int x = 0; x + windowSize - 1 < image.width; x += stepSize) {
                for (int y = 0; y + windowSize - 1 < image.height; y += stepSize) {
                    float[] featureVec = new float[windowSize * windowSize];
                    // Flattening the window into a vector
                    for (int dy = 0; dy < windowSize; dy++) {
                        System.arraycopy(image.pixels[y + dy], x, featureVec, dy * windowSize, windowSize);
                    }

                    output.add(new FloatKeypoint(0, 0, 0, 0, featureVec));
                }
            }

            return output;
        }
    }

    static class MyExtractor implements FeatureExtractor<SparseIntFV, FImage> {
        MyEngine engine;
        HardAssigner<float[], float[], IntFloatPair> assigner;

        public MyExtractor(HardAssigner<float[], float[], IntFloatPair> assigner, MyEngine engine) {
            this.engine = engine;
            this.assigner = assigner;
        }

        public SparseIntFV extractFeature(FImage image) {
            BagOfVisualWords<float[]> bovw = new BagOfVisualWords<>(this.assigner);
            // Return the historgram vector for the image
            return bovw.aggregate(engine.analyze(image));
        }
    }
}
