package uk.ac.soton.ecs.compvisgroup.apple.run1;


import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;

/**
 * OpenIMAJ Hello world!
 *
 */
public class App {
    public static void main( String[] args ) throws FileSystemException {
        KNearest knn = new KNearest();
        TinyImageProcessor ti = new TinyImageProcessor(16);

        // Open Dataset
        System.out.println("Loading Train and Test datasets");
        String resourcesDir = App.class.getClassLoader().getResource("").toString();
        // GroupDataset keeps the label for training data
        VFSGroupDataset<FImage> training = new VFSGroupDataset<>("zip:" + resourcesDir + "training.zip", ImageUtilities.FIMAGE_READER);
        // We don't need the label for the testing data, so we use a ListDataset
        VFSListDataset<FImage> testing = new VFSListDataset<>("zip:" + resourcesDir + "testing.zip", ImageUtilities.FIMAGE_READER);
        System.out.println(training);
        System.out.println(testing);


        for (FImage image : testing) {
            ti.generateTinyImage(image);
        }




    }


}
