package es.ugr.ic;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.sql.Timestamp;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import java.util.Scanner;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer.PoolingType;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
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
		int numEpochs = 2; // number of epochs to perform
		double learningRate = 0.006; //Learning rate
		
		SimpleDateFormat dateFormat = new SimpleDateFormat("HH.mm_dd.MM.yyyy");
		Date now = new Date();
		
		File resultsFile = new File("./data/results"+dateFormat.format(now)+".txt");
		File trainingStats = new File("./data/stats"+dateFormat.format(now)+".dl4f");
		
		// Get the DataSetIterators:
		DataSetIterator mnistTrain =  new MnistDataSetIterator(batchSize, MnistDataFetcher.NUM_EXAMPLES, false, true, false, rngSeed);
		DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, MnistDataFetcher.NUM_EXAMPLES_TEST, false, false, false, rngSeed);
		
		long tIni, tFinTrain, tFinEvTrain, tFinEvTest = 0;
		
		Scanner sc = new Scanner(System.in);
		String opcion = "";
		boolean salir = false;
		MultiLayerNetwork model;
		Evaluation eval;
		StatsListener stListener, stFileListener;
		boolean guardadoAcabado =false;
		
		wipeOutputArrays();

		do {
			System.out.println(SEPARADOR);
			System.out.println("Red neuronal artificial con DeepLearning4J");
			System.out.println("Elija una opcion:");
			System.out.println("\t1) Entrenar una red multicapa (capa de entrada , capa oculta ReLU y capa de salida softmax)");
			System.out.println("\t2) Entrenar una red basada en LeNet5 (capas convolutivas, capas pooling, capas ocultas densas y capa de salida softmax)");
			System.out.println("\t3) Cargar una red ya entrenada para evaluarla");
			System.out.println("\t4) Salir");
			System.out.print("Opción: ");
			opcion = sc.nextLine().trim().toLowerCase();
			System.out.println(SEPARADOR);

			switch (opcion) {
				
			case "1":
			case "entrenar una red multicapa":
				
				
				//learningRate = 0.01;
				
				model = createModelMultiLayer(numRows, numColumns, outputNum, rngSeed, learningRate);
				
				
				stListener = enableUI();
				stFileListener= saveStats(trainingStats);
				tIni = trainModel(numEpochs, mnistTrain, model, stListener, stFileListener);

				tFinTrain = System.currentTimeMillis();
				
				
				log.info("\n"+SEPARADOR+"\n");
				log.info("Evaluación sobre el conjunto de entrenamiento");
				
				//Reiniciar conjunto de entrenamiento
				mnistTrain.reset();
				
				//Evaluar sobre conjunto de entrenamiento
				eval = testModel(outputNum, mnistTrain, model);
				
				tFinEvTrain = System.currentTimeMillis();
				printEvaluationResults(tIni, tFinTrain, tFinEvTrain, eval, true);
				
				saveEvaluationResults(tIni, tFinTrain, tFinEvTrain, eval, true, resultsFile, model);
				wipeOutputArrays();
				
				
				log.info("\n"+SEPARADOR+"\n");
				log.info("Evaluación sobre el conjunto de test");
				eval = testModel(outputNum, mnistTest, model);
				
				tFinEvTest = System.currentTimeMillis();
				
				//Para medir el tiempo de evaluación de esta segunda evaluacion
				//se suma al punto de partida de la evaluacion (tFinTrain) la
				//diferencia entre las finalizaciones con lo que nos dará el t
				//que hubiera tenido la evaluación si hubiera ido primero
				printEvaluationResults(tIni, tFinTrain, (tFinTrain + (tFinEvTest- tFinEvTrain)), eval, false);
				saveEvaluationResults(tIni, tFinTrain, (tFinTrain + (tFinEvTest- tFinEvTrain)), eval, false, resultsFile, model);
				
				guardadoAcabado =false;
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
						guardadoAcabado= true;
						break;
					
					case "n":
					case "no":
						guardadoAcabado = true;
						break;
	
					default:
						System.err.println("\n\nOpción incorrecta\n\n");						
						break;
					}
				} while (guardadoAcabado == false);
				
				salir = true;
				break;

			case "2":
			case "entrenar una red basada en lenet5 ":
				
				
				learningRate = 0.013;
				batchSize=64;
				numEpochs=6;
				
				
				model = createModelConvolution(numRows, numColumns, outputNum, rngSeed, learningRate);
				
			
				stListener = enableUI();
				stFileListener= saveStats(trainingStats);
				tIni = trainAndEvalModel(numEpochs, outputNum, mnistTrain, mnistTest, model, stListener, stFileListener);

				tFinTrain = System.currentTimeMillis();
				
				
				log.info("\n"+SEPARADOR+"\n");
				log.info("Evaluación sobre el conjunto de entrenamiento");
				
				//Reiniciar conjunto de entrenamiento
				mnistTrain.reset();
				
				//Evaluar sobre conjunto de entrenamiento
				eval = testModel(outputNum, mnistTrain, model);
				
				tFinEvTrain = System.currentTimeMillis();
				printEvaluationResults(tIni, tFinTrain, tFinEvTrain, eval, true);
				
				saveEvaluationResults(tIni, tFinTrain, tFinEvTrain, eval, true, resultsFile, model);
				wipeOutputArrays();
				
				
				log.info("\n"+SEPARADOR+"\n");
				log.info("Evaluación sobre el conjunto de test");
				eval = testModel(outputNum, mnistTest, model);
				
				tFinEvTest = System.currentTimeMillis();
				
				//Para medir el tiempo de evaluación de esta segunda evaluacion
				//se suma al punto de partida de la evaluacion (tFinTrain) la
				//diferencia entre las finalizaciones con lo que nos dará el t
				//que hubiera tenido la evaluación si hubiera ido primero
				printEvaluationResults(tIni, tFinTrain, (tFinTrain + (tFinEvTest- tFinEvTrain)), eval, false);
				saveEvaluationResults(tIni, tFinTrain, (tFinTrain + (tFinEvTest- tFinEvTrain)), eval, false, resultsFile, model);
				
				guardadoAcabado =true;
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
						guardadoAcabado= true;
						break;
					
					case "n":
					case "no":
						guardadoAcabado = true;
						break;
	
					default:
						System.err.println("\n\nOpción incorrecta\n\n");						
						break;
					}
				} while (guardadoAcabado == false);
				
				salir = true;
				break;

			case "3":
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
				
				wipeOutputArrays();
				log.info("\n"+SEPARADOR+"\n");
				log.info("Evaluación sobre el conjunto de test");
				eval = testModel(outputNum, mnistTest, model);
				
				tFinEvTest = System.currentTimeMillis();
				
				//Para medir el tiempo de evaluación de esta segunda evaluacion
				//se suma al punto de partida de la evaluacion (tFinTrain) la
				//diferencia entre las finalizaciones con lo que nos dará el t
				//que hubiera tenido la evaluación si hubiera ido primero
				printEvaluationResults(tIni, tFinTrain, (tFinTrain + (tFinEvTest- tFinEvTrain)), eval, false);
				
				checkLabelsAndResults();
				
				salir = true;
				break;
				
			case "4":
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


	
	private static MultiLayerNetwork createModelMultiLayer(final int numRows, final int numColumns, int outputNum, int rngSeed, double learningRate) {
		log.info("Build model....");
		
		// include a random seed for reproducibility
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(rngSeed) 
				// use stochastic gradient descent as an optimization algorithm
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.iterations(1).learningRate(learningRate) // specify the learning rate
				.updater(Updater.NESTEROVS).momentum(0.9) // specify the rate of change of the learning rate.
				.regularization(true).l2(1e-4).list()
				.layer(0, new DenseLayer.Builder() // create the first, input layer with xavier initialization
						.nIn(numRows * numColumns)
						.nOut(1000)
						.activation("relu")
						.weightInit(WeightInit.XAVIER)
						.build())
				// create hidden layer
				.layer(1, new OutputLayer.Builder(LossFunction.MSE) 
						.nIn(1000)
						.nOut(outputNum)
						.activation("softmax")
						.weightInit(WeightInit.XAVIER)
						.build())
				.pretrain(false).backprop(true) // use backpropagation to adjust weights
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		return model;
	}
	
	private static MultiLayerNetwork createModelConvolution(final int numRows, final int numColumns, int outputNum, int rngSeed, double learningRate) {
		log.info("Build model....");
		
		   MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
	                .seed(rngSeed)
	                .iterations(1) // Training iterations as above
	                .regularization(true).l2(0.0005)
	                /*
	                    Uncomment the following for learning decay and bias
	                 */
	                .learningRate(learningRate)//.biasLearningRate(0.02)
	                //.learningRateDecayPolicy(LearningRatePolicy.Inverse).lrPolicyDecayRate(0.001).lrPolicyPower(0.75)
	                .weightInit(WeightInit.XAVIER)
	                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
	                //.updater(Updater.NESTEROVS).momentum(0.9)
	                .updater(Updater.ADADELTA).epsilon(1e-6)
	                .list()
	                .layer(0, new ConvolutionLayer.Builder(5, 5)
	                        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
	                        .nIn(1)
	                        .stride(1, 1)
	                        .nOut(20)
	                        .activation("identity")
	                        .build())
	                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
	                        .kernelSize(2,2)
	                        .stride(2,2)
	                        .build())
	                .layer(2, new ConvolutionLayer.Builder(5, 5)
	                        //Note that nIn need not be specified in later layers
	                        .stride(1, 1)
	                        .nOut(50)
	                        .activation("identity")
	                        .build())
	                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
	                        .kernelSize(2,2)
	                        .stride(2,2)
	                        .build())
	                .layer(4, new DenseLayer.Builder().activation("relu")
	                        .nOut(120).build())
	                .layer(5, new DenseLayer.Builder().activation("tanh")
	                        .nOut(84).build())
	                .layer(6, new DenseLayer.Builder().activation("sigmoid")
	                		.nOut(48).build())
	                .layer(7, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
	                        .nOut(outputNum)
	                        .activation("softmax")
	                        .build())
	                .setInputType(InputType.convolutionalFlat(28,28,1)) 
	                .backprop(true).pretrain(false).build();
		   
		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		return model;
	}
	
	private static MultiLayerNetwork createModelSimple(final int numRows, final int numColumns, int outputNum, int rngSeed, double learningRate) {
		log.info("Build model....");
		
		// include a random seed for reproducibility
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(rngSeed) 
				// use stochastic gradient descent as an optimization algorithm
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.iterations(1).learningRate(learningRate) // specify the learning rate
				.list()
				.layer(0, new OutputLayer.Builder(LossFunction.MSE) 
						.nIn(numRows * numColumns)
						.nOut(outputNum)
						.activation("softmax")
						.weightInit(WeightInit.XAVIER)
						.build())
				.pretrain(false).backprop(true) // use backpropagation to adjust weights
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		return model;
	}
	
	private static long trainModel(int numEpochs, DataSetIterator mnistTrain, MultiLayerNetwork model, StatsListener stListener, StatsListener stFileListener) {
		long tIni;
		
		model.init();
		model.setListeners(stListener, stFileListener);

		log.info("Train model....");
		tIni = System.currentTimeMillis();
		for (int i = 0; i < numEpochs; i++) {
			model.fit(mnistTrain);
		}
		return tIni;
	}

	private static long trainAndEvalModel(int numEpochs, int outputNum, DataSetIterator mnistTrain,
			DataSetIterator mnistTest, MultiLayerNetwork model, StatsListener stListener,
			StatsListener stFileListener) {
		long tIni;
		
		double accuary = 0.985;
		model.init();
		model.setListeners(stListener, stFileListener);
		
		log.info("Train model....");
		tIni = System.currentTimeMillis();
		for (int i = 0; i < numEpochs; i++) {
			model.fit(mnistTrain);
			
			log.info("*** Completed epoch {} ***", i);

            log.info("Evaluate model....");
            Evaluation eval = new Evaluation(outputNum);
            while(mnistTest.hasNext()){
                DataSet ds = mnistTest.next();
                INDArray output = model.output(ds.getFeatureMatrix(), false);
                eval.eval(ds.getLabels(), output);

            }
            
            log.info(eval.stats());
            if (eval.accuracy() > accuary) {
				accuary = eval.accuracy();
				
				double error = 1 - accuary;
				

				File RNAFile = new File("lenet"+error+".zip");
				
				try {
					saveRNA(model, RNAFile);
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				log.info("Guardada red en "+RNAFile.getAbsolutePath());
			}
            mnistTest.reset();
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
	
	private static void checkLabelsAndResults() {
		
		//Load labels from a file produced directly from the official labels
		//of the MNIST database 
		//http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
		File referenceLabels = new File("./data/referenceLabels.txt");
		
		List<String> lines;
		try {
			lines = Files.readAllLines(referenceLabels.toPath());
			if (lines.size() != 1) {
				log.info("El fichero leido tiene más de una linea");
				return;
			}
			
			String refLabelsStr = lines.get(0);
			
			refLabelsStr = refLabelsStr.substring(1, refLabelsStr.length()-1);
			
			System.out.println("reference labels sin corchetes: "+refLabelsStr);
			
			//Delete all whitespaces and non visible characters
			refLabelsStr = refLabelsStr.replaceAll("\\s", "");
			
			System.out.println("reference labels sin espacios: "+refLabelsStr);
			
			String[] singleLabels = refLabelsStr.split(",");
			
			int [] MNISTLabels= new int[singleLabels.length];
			
			for (int i = 0; i < singleLabels.length; i++) {
				MNISTLabels[i] = Integer.parseInt(singleLabels[i]);
			}
			
			log.info("Comparación de labels oficiales y de DL4J");
			if (MNISTLabels.length != labelsInt[0].length) {
				log.info("La longitud de los arrays de etiquetas es distinta");
				return;
			}
			boolean iguales = true;
			for (int i = 0; i < MNISTLabels.length; i++) {
				if (MNISTLabels[i] != labelsInt[0][i]) {
					log.info("La etiqueta "+i+" es distinta, en labels oficiales "
							+MNISTLabels[i]+" y en labels de MNISTDataSetIterator "+labelsInt[0][i]);
					iguales=false;
				}
			}
			
			if (iguales == false) {
				return;
			}
			else {
				log.info("labels oficiales igual a los de MNISTDataSetIterator");
			}
			
			log.info("Comparación de labels oficiales con resultados de la predicción");
			
			int errores = 0;
			for (int i = 0; i < MNISTLabels.length; i++) {
				if (MNISTLabels[i] != outputValuesInt[0][i]) {
					log.info("La predicción "+i+" es distinta, en labels oficiales "
							+MNISTLabels[i]+" y en la predicción "+outputValuesInt[0][i]);
					errores++;
				}
			}
			
			log.info("Número total de errores en la predicción: "+errores);
		} catch (IOException e) {
			log.error("Error leyendo los labels de "+referenceLabels.getAbsolutePath());
			e.printStackTrace();
		}
		
		
		
	}
	private static void printEvaluationResults(long tIni, long tFinTrain, long tFin, Evaluation eval, boolean train) {
	
		log.info("****************Resultados de la evaluación********************");
		log.info(eval.stats());
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
		


		log.info("Valores predichos vs etiquetas:");
		
		StringBuilder sbOut = new StringBuilder("Valores predichos: ");
		StringBuilder sbLab = new StringBuilder("Etiquetas: ");
		
		if (train) {
			for (int i = 0; i < 6; i++) {
				for (int j = 0; j < MnistDataFetcher.NUM_EXAMPLES_TEST; j++) {
					sbOut.append(outputValuesInt[i][j]);
					sbLab.append(labelsInt[i][j]);
				}
			}
		}
		else
		{
			for (int j = 0; j < MnistDataFetcher.NUM_EXAMPLES_TEST; j++) {
				sbOut.append(outputValuesInt[0][j]);
				sbLab.append(labelsInt[0][j]);
			}
		}
		log.info(sbOut.toString());
		log.info(sbLab.toString());
		
       
       
	}


	private static void wipeOutputArrays() {
		for (int i = 0; i < 6; i++) {
        	 Arrays.fill(outputValuesInt[i], 0);
             Arrays.fill(labelsInt[i], 0);
		}
	}

	private static void saveEvaluationResults(long tIni, long tFinTrain, long tFin, Evaluation eval, boolean train,
			File file, MultiLayerNetwork model) {

		FileWriter fwr = null;

		try {
			fwr = new FileWriter(file, true);
			
			Timestamp timestamp = new Timestamp(System.currentTimeMillis());
	        
			if (train) {
				fwr.write(
						"****************Resultados de la evaluación sobre el conjunto de entrenamiento********************\n");
			} else {
				fwr.write("****************Resultados de la evaluación sobre el conjunto de test********************\n");

			}
			fwr.write(timestamp+"\n\n");
			StringBuilder sbConf = new StringBuilder("Configuracion de la red: ");
			List<NeuralNetConfiguration> layers = model.getLayerWiseConfigurations().getConfs();
			for (int i = 0; i < layers.size(); i++) {
				
				sbConf.append("\nLayer "+i+" "+layers.get(i).toJson());
			}
			fwr.write(sbConf.toString());
			fwr.write(eval.stats()+"\n");

		
			long errores = 0;

			if (train) {
				for (int i = 0; i < 6; i++) {
					for (int j = 0; j < MnistDataFetcher.NUM_EXAMPLES_TEST; j++) {
						if (labelsInt[i][j] != outputValuesInt[i][j]) {
							errores++;
						}
					}
				}
			} else {
				for (int j = 0; j < MnistDataFetcher.NUM_EXAMPLES_TEST; j++) {
					if (labelsInt[0][j] != outputValuesInt[0][j]) {
						errores++;
					}
				}
			}

			double errorTotal = 0;
			if (train) {
				errorTotal = ((double) errores / MnistDataFetcher.NUM_EXAMPLES) * 100;
			} else {
				errorTotal = ((double) errores / MnistDataFetcher.NUM_EXAMPLES_TEST) * 100;
			}
			fwr.write("Error total: " + errorTotal + "%\n");

			
			fwr.write("Valores predichos vs etiquetas:\n");
			StringBuilder sbOut = new StringBuilder("Valores predichos: ");
			StringBuilder sbLab = new StringBuilder("Etiquetas: ");
			
			if (train) {
				for (int i = 0; i < 6; i++) {
					for (int j = 0; j < MnistDataFetcher.NUM_EXAMPLES_TEST; j++) {
						sbOut.append(outputValuesInt[i][j]);
						sbLab.append(labelsInt[i][j]);
					}
				}
			}
			else
			{
				for (int j = 0; j < MnistDataFetcher.NUM_EXAMPLES_TEST; j++) {
					sbOut.append(outputValuesInt[0][j]);
					sbLab.append(labelsInt[0][j]);
				}
			}
			
			fwr.write(sbOut.toString()+ "\n");
			fwr.write(sbLab.toString()+ "\n");
			
		

			fwr.write("\nTiempo total de entrenamiento: " + (tFinTrain - tIni) + " ms\n");
			fwr.write("Tiempo total de evaluación: " + (tFin - tFinTrain) + " ms\n");
			fwr.write("Tiempo total: " + (tFin - tIni) + " ms\n\n");

		} catch (IOException e) {

			e.printStackTrace();
		} finally {
			if (fwr != null) {
				try {
					fwr.flush();
					fwr.close();
				} catch (IOException e) {
				}

			}
		}

	}
	

	private static StatsListener enableUI() {
		//Initialize the user intedrface backend
        UIServer uiServer = UIServer.getInstance();
        
        //Configure where the network information (gradients, activations, score vs. time etc) is to be stored
        //Then add the StatsListener to collect this information from the network, as it trains
        //Alternative: new FileStatsStorage(File) - see UIStorageExample
        StatsStorage statsStorage = new InMemoryStatsStorage();
        
        StatsListener statListener = new StatsListener(statsStorage);
        
        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
        uiServer.attach(statsStorage);
        
        return statListener;
	}

	private static StatsListener saveStats(File stFile)
	{
		try {
			
			if (!stFile.exists()) {
				Files.createDirectories(stFile.getParentFile().toPath());
				
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		StatsStorage statsStorage = new FileStatsStorage(stFile);
		StatsListener statListener = new StatsListener(statsStorage);
		return statListener;
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
