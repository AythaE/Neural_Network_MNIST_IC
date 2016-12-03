package es.ugr.ic;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Scanner;

import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class RNA {
	
	
	private static Logger log = LoggerFactory.getLogger(RNA.class);
	private static final String SEPARADOR = "================================================================================";
	
	private static int[][] outputValuesInt = new int[6][MnistDataFetcher.NUM_EXAMPLES_TEST];
	private static int[][] labelsInt = new int[6][MnistDataFetcher.NUM_EXAMPLES_TEST];
	private static int columnIterator = 0, rowIterator = 0;
	private static int mostProbableOutput = 0, LabelInt = 0;
	
	
	public static void main(String[] args) throws Exception {
		// number of rows and columns in the input pictures
		final int numRows = 28;
		final int numColumns = 28;
		int outputNum = 10; // number of output classes
		int batchSize = 128; // batch size for each epoch
		int rngSeed = 123; // random number seed for reproducibility
		int numEpochs = 1; // number of epochs to perform
		
	
		// Get the DataSetIterators:
		DataSetIterator mnistTrain =  new MnistDataSetIterator(batchSize, MnistDataFetcher.NUM_EXAMPLES, false, true, false, rngSeed);
		DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, MnistDataFetcher.NUM_EXAMPLES_TEST, false, false, false, rngSeed);
		
		long tIni, tFinTrain, tFinEvTrain, tFinEvTest = 0;
		
		Scanner sc = new Scanner(System.in);
		String opcion = "";
		boolean salir = false;
		MultiLayerNetwork model;
		Evaluation eval;
		
		do {
			System.out.println(SEPARADOR);
			System.out.println("Red neuronal artificial con DeepLearning4J");
			System.out.println("Elija una opcion:");
			System.out.println("\t1) Entrenar una red desde 0");
			System.out.println("\t2) Cargar una red ya entrenada para evaluarla");
			System.out.println("\t3) Salir");
			System.out.print("Opción: ");
			opcion = sc.nextLine().trim().toLowerCase();
			System.out.println(SEPARADOR);

			switch (opcion) {
			case "1":
			case "entrenar":
			case "entrenar una red desde 0":
				
				model = createModel(numRows, numColumns, outputNum, rngSeed);
				
				tIni = trainModel(numEpochs, mnistTrain, model);

				tFinTrain = System.currentTimeMillis();
				
				
				log.info("\n"+SEPARADOR+"\n");
				log.info("Evaluación sobre el conjunto de entrenamiento");
				
				//Reiniciar conjunto de entrenamiento
				mnistTrain =  new MnistDataSetIterator(batchSize, MnistDataFetcher.NUM_EXAMPLES, false, true, false, rngSeed);
				
				//Evaluar sobre conjunto de entrenamiento
				eval = testModel(outputNum, mnistTrain, model);
				
				tFinEvTrain = System.currentTimeMillis();
				printEvaluationResults(tIni, tFinTrain, tFinEvTrain, eval, true);
				
				log.info("\n"+SEPARADOR+"\n");
				log.info("Evaluación sobre el conjunto de test");
				eval = testModel(outputNum, mnistTest, model);
				
				tFinEvTest = System.currentTimeMillis();
				
				//Para medir el tiempo de evaluación de esta segunda evaluacion
				//se suma al punto de partida de la evaluacion (tFinTrain) la
				//diferencia entre las finalizaciones con lo que nos dará el t
				//que hubiera tenido la evaluación si hubiera ido primero
				printEvaluationResults(tIni, tFinTrain, (tFinTrain + (tFinEvTest- tFinEvTrain)), eval, false);
				
				boolean acabado =false;
				Thread.sleep(500);
				do{
					System.out.print("¿Desea guardar la red entrenada? (Si/No): ");
					String guardar = sc.nextLine().trim().toLowerCase();
					switch (guardar) {
					case "s":
					case "si":
						System.out.println("La red se guardará en formato .zip");
						System.out.print("Introduzca la ruta del fichero donde desea guardar la red (absoluta o relativa): ");
						String filePath = sc.nextLine().trim();
						
						try {
							saveRNA(model, new File(filePath));
						} catch (Exception e) {
							System.err.println("Error guardando la red, puede ver los detalles a continuación");
							e.printStackTrace();
							return;
						}
						
						System.out.println("Red neuronal guardada correctamente en "+filePath);
						acabado= true;
						break;
					
					case "n":
					case "no":
						acabado = true;
						break;
	
					default:
						System.err.println("\n\nOpción incorrecta\n\n");						
						break;
					}
				} while (acabado == false);
				
				salir = true;
				break;

			case "2":
			case "cargar":
			case "cargar una red ya entrenada para evaluarla":
				
				System.out.print("Introduzca la ruta del fichero donde este guardada la red (absoluta o relativa): ");
				String filePath = sc.nextLine().trim();
				
				File modelFile = new File(filePath);
				
				if (!modelFile.exists()) {
					System.err.println("\n\nEl fichero "+ filePath +" no existe");
					break;
				}
				
				try {
					model = loadRNA(modelFile);
				} catch (Exception e) {
					System.err.println("Error cargando la red, puede ver los detalles a continuación");
					e.printStackTrace();
					return;
				}
				
				
				
				tFinTrain = tIni= System.currentTimeMillis();
				
				log.info("\n"+SEPARADOR+"\n");
				log.info("Evaluación sobre el conjunto de entrenamiento");
				//Evaluar sobre conjunto de entrenamiento
				eval = testModel(outputNum, mnistTrain, model);
				
				tFinEvTrain = System.currentTimeMillis();
				printEvaluationResults(tIni, tFinTrain, tFinEvTrain, eval, true);
				
				log.info("\n"+SEPARADOR+"\n");
				log.info("Evaluación sobre el conjunto de test");
				eval = testModel(outputNum, mnistTest, model);
				
				tFinEvTest = System.currentTimeMillis();
				
				//Para medir el tiempo de evaluación de esta segunda evaluacion
				//se suma al punto de partida de la evaluacion (tFinTrain) la
				//diferencia entre las finalizaciones con lo que nos dará el t
				//que hubiera tenido la evaluación si hubiera ido primero
				printEvaluationResults(tIni, tFinTrain, (tFinTrain + (tFinEvTest- tFinEvTrain)), eval, false);
				
				salir = true;
				break;
				
			case "3":
			case "salir":
				salir=true;
				break;
			default:
				System.err.println("\n\nOpción incorrecta, las opciones permitidas son: ");
				System.err.println("Para la primera opción: 1, entrenar y entrenar una red desde 0 ");
				System.err
						.println("Para la segunda opción: 2, cargar y cargar una red ya entrenada para evaluarla\n\n");
				Thread.sleep(500);
				break;
			}
		} while (salir == false);
		sc.close();
		return;
	}

	
	private static MultiLayerNetwork createModel(final int numRows, final int numColumns, int outputNum, int rngSeed) {
		log.info("Build model....");
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(rngSeed) // include
																							// a
																							// random
																							// seed
																							// for
																							// reproducibility
				// use stochastic gradient descent as an optimization algorithm
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1).learningRate(0.006) // specify
																														// the
																														// learning
																														// rate
				.updater(Updater.NESTEROVS).momentum(0.9) // specify the rate of
															// change of the
															// learning rate.
				.regularization(true).l2(1e-4).list()
				.layer(0, new DenseLayer.Builder() // create the first, input
													// layer with xavier
													// initialization
						.nIn(numRows * numColumns).nOut(1000).activation("relu").weightInit(WeightInit.XAVIER).build())
				.layer(1,
						new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD) // create
																					// hidden
																					// layer
								.nIn(1000).nOut(outputNum).activation("softmax").weightInit(WeightInit.XAVIER).build())
				.pretrain(false).backprop(true) // use backpropagation to adjust
												// weights
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		return model;
	}

	
	private static long trainModel(int numEpochs, DataSetIterator mnistTrain, MultiLayerNetwork model) {
		long tIni;
		
		model.init();
		// print the score with every 1 iteration
		model.setListeners(new ScoreIterationListener(5));

		log.info("Train model....");
		tIni = System.currentTimeMillis();
		for (int i = 0; i < numEpochs; i++) {
			model.fit(mnistTrain);
		}
		return tIni;
	}
	
	private static Evaluation testModel(int outputNum, DataSetIterator mnistTest, MultiLayerNetwork model) {
		log.info("Evaluate model....");
		Evaluation eval = new Evaluation(outputNum); // create an evaluation
														// object with 10
														// possible classes
		
		
		columnIterator = 0; rowIterator = 0; 
		
		while (mnistTest.hasNext()) {
			DataSet next = mnistTest.next();
			INDArray output = model.output(next.getFeatureMatrix()); // get the
																		// networks
																		// prediction
			INDArray labels = next.getLabels();
			
			
			for (int i = 0; i < output.rows(); i++) {
				INDArray outputProb = output.getRow(i);
				INDArray labelsProb = labels.getRow(i);
				mostProbableOutput = 0;
				LabelInt = 0;

				for (int j = 1; j < outputProb.columns(); j++) {
					if (outputProb.getDouble(mostProbableOutput) < outputProb.getDouble(j)) {
						mostProbableOutput = j;
					}
					if (labelsProb.getDouble(LabelInt) < labelsProb.getDouble(j)) {
						LabelInt = j;
					}
				}
				
				//Si el iterador de la columna llega al numero total de datos
				//por fila es momento de cambiar de fila 
				if (columnIterator == MnistDataFetcher.NUM_EXAMPLES_TEST) {
					rowIterator++;
					columnIterator=0;
				}
				outputValuesInt[rowIterator][columnIterator] = mostProbableOutput;
				
				labelsInt[rowIterator][columnIterator] = LabelInt;
				columnIterator++;
			}
			eval.eval(next.getLabels(), output); // check the prediction against
													// the true class
		}
		return eval;
	}
	
	private static void printEvaluationResults(long tIni, long tFinTrain, long tFin, Evaluation eval, boolean train) {
		log.info(eval.stats());
		log.info("****************Example finished********************");

		log.info("Tiempo total de entrenamiento: " + (tFinTrain - tIni) + " ms");
		log.info("Tiempo total de evaluación: " + (tFin - tFinTrain) + " ms");
		log.info("Tiempo total: " + (tFin - tIni) + " ms");
		long errores = 0;
		
		if (train) {
			for (int i = 0; i < 6; i++) {
				for (int j = 0; j < MnistDataFetcher.NUM_EXAMPLES_TEST; j++) {
					if (labelsInt[i][j] != outputValuesInt[i][j]) {
						errores++;
					}
				}
			}
		}
		else{
			for (int j = 0; j < MnistDataFetcher.NUM_EXAMPLES_TEST; j++) {
				if (labelsInt[0][j] != outputValuesInt[0][j]) {
					errores++;
				}
			}
		}
	
		double errorTotal=0;
		if (train) {
			errorTotal = ((double) errores / MnistDataFetcher.NUM_EXAMPLES) * 100;
		}
		else {
			errorTotal = ((double) errores / MnistDataFetcher.NUM_EXAMPLES_TEST) * 100;
		}
		log.info("Error total: " + errorTotal + "%");
		

		if (train==false) {
			log.info("Valores predichos vs etiquetas:");
			System.err.println(Arrays.toString(outputValuesInt[0]));
	        System.err.println(Arrays.toString(labelsInt[0]));
		}
		
        for (int i = 0; i < 6; i++) {
        	 Arrays.fill(outputValuesInt[i], 0);
             Arrays.fill(labelsInt[i], 0);
		}
       
	}

	

	

	
	private static void saveRNA(MultiLayerNetwork model, File modelFile) throws IOException {

		boolean saveUpdater = true; // Updater: i.e., the state for Momentum,
									// RMSProp, Adagrad etc. Save this if you
									// want to train your network more in the
									// future

		ModelSerializer.writeModel(model, modelFile, saveUpdater);
	}

	private static MultiLayerNetwork loadRNA(File modelFile) throws IOException {

		MultiLayerNetwork restored = ModelSerializer.restoreMultiLayerNetwork(modelFile);
		return restored;
	}

}
