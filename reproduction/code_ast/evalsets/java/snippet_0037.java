package org.tigr.microarray.mev.persistence;

import java.beans.Encoder;
import java.beans.Expression;
import java.beans.PersistenceDelegate;
import java.io.File;
import java.io.IOException;

import org.tigr.microarray.mev.MultipleArrayViewer;
import org.tigr.microarray.mev.cluster.ClusterWrapper;

public class ClusterWrapperPersistenceDelegate extends PersistenceDelegate {

	@Override
	public Expression instantiate(Object oldInstance, Encoder encoder) {
		ClusterWrapper pa = (ClusterWrapper) oldInstance;
		if(pa != null && pa.getClusters() != null && pa.getClusters()[0] != null) {
			try {
				File outputFile = File.createTempFile("clusterwrapper", ".bin", new File(MultipleArrayViewer.CURRENT_TEMP_DIR));
		        outputFile.deleteOnExit();
		        PersistenceObjectFactory.writeIntMatrix(outputFile, pa.getClusters());
		        return new Expression((ClusterWrapper) oldInstance, new PersistenceObjectFactory().getClass(), "readIntMatrix",
						new Object[]{outputFile.getName()});
			} catch (IOException ioe){
				System.out.println("Can't write to file to save clusters");
				return null;
			} 
		} else {
			return new Expression((ClusterWrapper) oldInstance, ClusterWrapper.class, "wrapClusters",
					new Object[]{null});
		}
	}


}