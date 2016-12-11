/*
 * Archivo: RNA.java 
 * Proyecto: RNA_DL4J_MNIST
 * 
 * Autor: Aythami Estévez Olivas
 * Email: aythae@correo.ugr.es
 * Fecha: 09-dic-2016
 * Asignatura: Inteligencia Computacional
 */
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

/**
 * The Class RNA.
 */
public class RNA {
	
	
	/** The log. */
	private static Logger log = LoggerFactory.getLogger(RNA.class);
	
	/**The Constant SEPARADOR used for a cleaner presentation of info in the console.*/
	private static final String SEPARADOR = "================================================================================";
	
	/** 
	 * The output values int matrix, each row represents 10000 int numbers, this
	 * is a matrix to store the outputs of the evaluations. The training set
	 * (of 60000 examples) forces it to be a matrix. Initially this was one-
	 * dimensional array, but the size of the training set overflow the int max
	 * number so a single index could not handle the access to the entire array. 
	 */
	private static int[][] outputValuesInt = new int[6][MnistDataFetcher.NUM_EXAMPLES_TEST];
	
	/** 
	 * The labels int matrix, each row represents 10000 int numbers, this
	 * is a matrix for the same reason as outputValuesInt.
	 */
	private static int[][] labelsInt = new int[6][MnistDataFetcher.NUM_EXAMPLES_TEST];
	
	/** The row iterator. */
	private static int columnIterator = 0, rowIterator = 0;
	
	/** The Label int. */
	private static int mostProbableOutput = 0, LabelInt = 0;
	
	/** The Scanner to handle user input. */
	private static Scanner sc = new Scanner(System.in);
	
	/** The date format. */
	private static SimpleDateFormat dateFormat = new SimpleDateFormat("HH.mm_dd.MM.yyyy");
	
	/** Date used to create a timestamp for the files. */
	private static Date now = new Date();
	
	/** The results file. */
	private static File resultsFile = new File("./data/results"+dateFormat.format(now)+".txt");
	
	/** The training stats file. */
	private static File trainingStats = new File("./data/stats"+dateFormat.format(now)+".dl4f");

	
	/**
	 * The main method.
	 *
	 * @param args the arguments
	 * @throws Exception the exception
	 */
	public static void main(String[] args) throws Exception {
		// number of rows and columns in the input pictures
		final int numRows = 28;
		final int numColumns = 28;
		int outputNum = 10; // number of output classes
		int batchSize = 128; // batch size for each epoch
		int rngSeed = 123; // random number seed for reproducibility
		int numEpochs = 2; // number of epochs to perform
		double learningRate = 0.006; //Learning rate
		
		
		
		// Get the DataSetIterators:
		DataSetIterator mnistTrain =  new MnistDataSetIterator(batchSize, MnistDataFetcher.NUM_EXAMPLES, false, true, false, rngSeed);
		DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, MnistDataFetcher.NUM_EXAMPLES_TEST, false, false, false, rngSeed);
		
		long tIni, tFinTrain, tFinEvTrain, tFinEvTest = 0;
		
		String opcion = "";
		boolean salir = false;
		MultiLayerNetwork model;
		Evaluation eval;
		StatsListener stListener, stFileListener;
		boolean guardadoAcabado =false;
		boolean periodicSave = false;
		wipeOutputArrays();

		do {
			System.out.println(SEPARADOR);
			System.out.println("Red neuronal artificial con DeepLearning4J");
			System.out.println("Elija una opcion:");
			System.out.println("\t1) Entrenar una red multicapa (capa de entrada , capa oculta tanh y capa de salida softmax)");
			System.out.println("\t2) Entrenar una red basada en LeNet5 (capas convolutivas, capas pooling, capas ocultas densas y capa de salida softmax)");
			System.out.println("\t3) Cargar una red ya entrenada para evaluarla");
			System.out.println("\t4) Cargar una red ya entrenada para seguir entrenandola");
			System.out.println("\t5) Salir");
			System.out.print("Opción: ");
			opcion = sc.nextLine().trim().toLowerCase();
			System.out.println(SEPARADOR);

			switch (opcion) {
				
			case "1":
			case "entrenar una red multicapa":
				
				//Change the default params
				numEpochs=32;
				learningRate = 0.0775;
				
				
				model = createModelMultiLayer(numRows, numColumns, outputNum, rngSeed, learningRate);
				
				
				stListener = enableUI();
				stFileListener= saveStats(trainingStats);
				
				//Para entrenar y evaluar en cada epoch
				//periodicSave = true;
				//tIni = trainAndEvalModel(numEpochs, outputNum, mnistTrain, mnistTest, model, stListener, stFileListener, periodicSave);
				
				tIni = trainModel(numEpochs, mnistTrain, model, stListener, stFileListener);
				
				tFinTrain = System.currentTimeMillis();
				
				
				log.info("\n"+SEPARADOR+"\n");
				log.info("Evaluación sobre el conjunto de entrenamiento");
				
				//Reiniciar conjunto de entrenamiento
				mnistTrain.reset();
				mnistTest.reset();
				//Evaluar sobre conjunto de entrenamiento
				eval = testModel(outputNum, mnistTrain, model);
				
				tFinEvTrain = System.currentTimeMillis();
				printResults(tIni, tFinTrain, tFinEvTrain, eval, true);
				
				saveResults(tIni, tFinTrain, tFinEvTrain, eval, true, resultsFile, model);
				wipeOutputArrays();
				
				
				log.info("\n"+SEPARADOR+"\n");
				log.info("Evaluación sobre el conjunto de test");
				eval = testModel(outputNum, mnistTest, model);
				
				tFinEvTest = System.currentTimeMillis();
				
				//Para medir el tiempo de evaluación de esta segunda evaluacion
				//se suma al punto de partida de la evaluacion (tFinTrain) la
				//diferencia entre las finalizaciones con lo que nos dará el t
				//que hubiera tenido la evaluación si hubiera ido primero
				printResults(tIni, tFinTrain, (tFinTrain + (tFinEvTest- tFinEvTrain)), eval, false);
				saveResults(tIni, tFinTrain, (tFinTrain + (tFinEvTest- tFinEvTrain)), eval, false, resultsFile, model);
				
				
				//Save the network if the user wants 
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
			case "entrenar una red basada en lenet5":
				
				//Change the default params
				learningRate = 0.013;
				batchSize=64;
				numEpochs=22;
				
				periodicSave=true;
				
				model = createModelConvolution(numRows, numColumns, outputNum, rngSeed, learningRate);
				
			
				stListener = enableUI();
				stFileListener= saveStats(trainingStats);
				
				//Para entrenar y evaluar en cada epoch
				//periodicSave = true;
				//tIni = trainAndEvalModel(numEpochs, outputNum, mnistTrain, mnistTest, model, stListener, stFileListener, periodicSave);
				
				tIni = trainModel(numEpochs, mnistTrain, model, stListener, stFileListener);
				
				tFinTrain = System.currentTimeMillis();
				
				
				log.info("\n"+SEPARADOR+"\n");
				log.info("Evaluación sobre el conjunto de entrenamiento");
				
				//Reiniciar conjunto de entrenamiento
				mnistTrain.reset();
				
				//Evaluar sobre conjunto de entrenamiento
				eval = testModel(outputNum, mnistTrain, model);
				
				tFinEvTrain = System.currentTimeMillis();
				printResults(tIni, tFinTrain, tFinEvTrain, eval, true);
				
				saveResults(tIni, tFinTrain, tFinEvTrain, eval, true, resultsFile, model);
				wipeOutputArrays();
				
				
				log.info("\n"+SEPARADOR+"\n");
				log.info("Evaluación sobre el conjunto de test");
				eval = testModel(outputNum, mnistTest, model);
				
				tFinEvTest = System.currentTimeMillis();
				
				//Para medir el tiempo de evaluación de esta segunda evaluacion
				//se suma al punto de partida de la evaluacion (tFinTrain) la
				//diferencia entre las finalizaciones con lo que nos dará el t
				//que hubiera tenido la evaluación si hubiera ido primero
				printResults(tIni, tFinTrain, (tFinTrain + (tFinEvTest- tFinEvTrain)), eval, false);
				saveResults(tIni, tFinTrain, (tFinTrain + (tFinEvTest- tFinEvTrain)), eval, false, resultsFile, model);
				

				//Save the network if the user wants 
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

			case "3":
			case "cargar una red ya entrenada para evaluarla":
				
				//Load the NN from the file specified by the user
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
				
				//Prints the network config
				log.info(getNetConfigurationAsString(model));
				
				tFinTrain = tIni= System.currentTimeMillis();
				
				log.info("\n"+SEPARADOR+"\n");
				log.info("Evaluación sobre el conjunto de entrenamiento");
				//Evaluar sobre conjunto de entrenamiento
				eval = testModel(outputNum, mnistTrain, model);
				
				tFinEvTrain = System.currentTimeMillis();
				printResults(tIni, tFinTrain, tFinEvTrain, eval, true);
				
				wipeOutputArrays();
				log.info("\n"+SEPARADOR+"\n");
				log.info("Evaluación sobre el conjunto de test");
				eval = testModel(outputNum, mnistTest, model);
				
				tFinEvTest = System.currentTimeMillis();
				
				//Para medir el tiempo de evaluación de esta segunda evaluacion
				//se suma al punto de partida de la evaluacion (tFinTrain) la
				//diferencia entre las finalizaciones con lo que nos dará el t
				//que hubiera tenido la evaluación si hubiera ido primero
				printResults(tIni, tFinTrain, (tFinTrain + (tFinEvTest- tFinEvTrain)), eval, false);
				
				checkLabelsAndResults();
				
				salir = true;
				break;
				
			case "4":
			case "cargar una red ya entrenada para seguir entrenandola":
				
				//Change the default params
				learningRate = 0.013;
				batchSize=64;
				numEpochs=30;
				
				//Load the NN from the file specified by the user
				System.out.print("Introduzca la ruta del fichero donde este guardada la red (absoluta o relativa): ");
				String modelPath = sc.nextLine().trim();
				
				File modelF = new File(modelPath);
				
				if (!modelF.exists()) {
					System.err.println("\n\nEl fichero "+ modelPath +" no existe");
					break;
				}
				
				try {
					model = loadRNA(modelF);
				} catch (Exception e) {
					System.err.println("Error cargando la red, puede ver los detalles a continuación");
					e.printStackTrace();
					return;
				}
				
				//Enable training UI
				stListener = enableUI();
				stFileListener= saveStats(trainingStats);
				
				
				tIni = trainAndEvalModel(numEpochs, outputNum, mnistTrain, mnistTest, model, stListener, stFileListener, true);

				tFinTrain = System.currentTimeMillis();
				
				
				log.info("\n"+SEPARADOR+"\n");
				log.info("Evaluación sobre el conjunto de entrenamiento");
				
				//Reiniciar conjunto de entrenamiento
				mnistTrain.reset();
				mnistTest.reset();
				//Evaluar sobre conjunto de entrenamiento
				eval = testModel(outputNum, mnistTrain, model);
				
				tFinEvTrain = System.currentTimeMillis();
				printResults(tIni, tFinTrain, tFinEvTrain, eval, true);
				
				saveResults(tIni, tFinTrain, tFinEvTrain, eval, true, resultsFile, model);
				wipeOutputArrays();
				
				
				log.info("\n"+SEPARADOR+"\n");
				log.info("Evaluación sobre el conjunto de test");
				eval = testModel(outputNum, mnistTest, model);
				
				tFinEvTest = System.currentTimeMillis();
				
				//Para medir el tiempo de evaluación de esta segunda evaluacion
				//se suma al punto de partida de la evaluacion (tFinTrain) la
				//diferencia entre las finalizaciones con lo que nos dará el t
				//que hubiera tenido la evaluación si hubiera ido primero
				printResults(tIni, tFinTrain, (tFinTrain + (tFinEvTest- tFinEvTrain)), eval, false);
				saveResults(tIni, tFinTrain, (tFinTrain + (tFinEvTest- tFinEvTrain)), eval, false, resultsFile, model);
				
				//Save the network if the user wants 
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
						filePath = sc.nextLine().trim();
						
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
				
				
			case "5":
			case "salir":
				salir=true;
				break;
			default:
				System.err.println("\n\nOpción incorrecta, las opciones permitidas son: ");
				System.err.println("Para la primera opción: 1 y entrenar una red multicapa");
				System.err.println("Para la segunda opción: 2 y entrenar una red basa en lenet5");
				System.err.println("Para la tercera opción: 3 y cargar una red ya entrenada para evaluarla");
				System.err.println("Para la cuarta  opción: 4 y cargar una red ya entrenada para seguir entrenandola");
				System.err.println("Para la quinta  opción: 5 y salir\n\n");
				Thread.sleep(500);
				break;
			}
		} while (salir == false);
		sc.close();
		return;
	}


	
	/**
	 * Creates the model for a multi layer NN.
	 *
	 * @param numRows the num rows of the input image
	 * @param numColumns the num columns of the input image
	 * @param outputNum the number of output classes
	 * @param rngSeed the rng seed
	 * @param learningRate the learning rate
	 * @return the multi layer network
	 */
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
				.layer(1, new OutputLayer.Builder(LossFunction.MCXENT) 
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
	
	/**
	 * Creates the model for convolution NN.
	 *
	 * @param numRows the num rows of the input image
	 * @param numColumns the num columns of the input image
	 * @param outputNum the number of output classes
	 * @param rngSeed the rng seed
	 * @param learningRate the learning rate
	 * @return the multi layer network
	 */
	private static MultiLayerNetwork createModelConvolution(final int numRows, final int numColumns, int outputNum, int rngSeed, double learningRate) {
		log.info("Build model....");
		
		   MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
	                .seed(rngSeed)
	                .iterations(1) // Training iterations as above
	                .regularization(true).l2(0.0005)
	                .learningRate(learningRate).weightInit(WeightInit.XAVIER)
	                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
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
	                .layer(6, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
	                        .nOut(outputNum)
	                        .activation("softmax")
	                        .build())
	                .setInputType(InputType.convolutionalFlat(28,28,1)) 
	                .backprop(true).pretrain(false).build();
		   
		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		return model;
	}
	
	/**
	 * Train model.
	 *
	 * @param numEpochs the num epochs
	 * @param mnistTrain the mnist train DataSetIterator
	 * @param model the NN model
	 * @param stListener the stats listener for the UI
	 * @param stFileListener the stats listener for the file 
	 * @return the initial time of the training
	 */
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

	/**
	 * Train and evaluate model after each epoch. This also could save the 
	 * network and its evaluation results each 10 epoch with the periodicSave
	 * flag. In addition it automatically safe the network when a evaluation
	 * has the better accuracy until now
	 *
	 * @param numEpochs the num epochs
	 * @param outputNum the output num
	 * @param mnistTrain the mnist train DataSetIterator
	 * @param mnistTest the mnist test DataSetIterator
	 * @param model the NN model
	 * @param stListener the stats listener for the UI
	 * @param stFileListener the stats listener for the file 
	 * @param periodicSave the periodic save flag
	 * @return the long
	 */
	private static long trainAndEvalModel(int numEpochs, int outputNum, DataSetIterator mnistTrain,
			DataSetIterator mnistTest, MultiLayerNetwork model, StatsListener stListener,
			StatsListener stFileListener, boolean periodicSave) {
		long tIni;
		
		double minAccuracy = 0.97, pastAccuracy= 0;
		
		int peorOIgual = 0;
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
            
            //If accuracy is higher than a minimum then save the NN 
            //If periodic save is enabled and this epoch is a multiple of 10
            //then save the results also
            if (eval.accuracy() > minAccuracy || (periodicSave && (i==1 || (i > 0 && (i+1) % 10 == 0)))) {
				double error;
            	
				if (eval.accuracy() > minAccuracy) {
					minAccuracy = eval.accuracy();
					
					error = 1 - minAccuracy;

				}
				else {

					error = 1 - eval.accuracy();
				}
				

				File RNAFile = new File("LeNet"+error+".zip");
				
				try {
					saveRNA(model, RNAFile);
				} catch (IOException e) {
					e.printStackTrace();
				}
				log.info("Guardada red en "+RNAFile.getAbsolutePath());
				
				if (periodicSave && (i==1 || (i > 0 && (i+1) % 10 == 0))) {
							
					long tFinTrain = System.currentTimeMillis();
					
					
					log.info("\n"+SEPARADOR+"\n");
					log.info("Evaluación sobre el conjunto de entrenamiento");
					
					//Reiniciar conjunto de entrenamiento
					mnistTrain.reset();
					mnistTest.reset();
					//Evaluar sobre conjunto de entrenamiento
					eval = testModel(outputNum, mnistTrain, model);
					
					long tFinEvTrain = System.currentTimeMillis();
					printResults(tIni, tFinTrain, tFinEvTrain, eval, true);
					
					saveResults(tIni, tFinTrain, tFinEvTrain, eval, true, resultsFile, model);
					wipeOutputArrays();
					
					
					
					log.info("\n"+SEPARADOR+"\n");
					log.info("Evaluación sobre el conjunto de test");
					eval = testModel(outputNum, mnistTest, model);
					
					long tFinEvTest = System.currentTimeMillis();
					
					//Para medir el tiempo de evaluación de esta segunda evaluacion
					//se suma al punto de partida de la evaluacion (tFinTrain) la
					//diferencia entre las finalizaciones con lo que nos dará el t
					//que hubiera tenido la evaluación si hubiera ido primero
					printResults(tIni, tFinTrain, (tFinTrain + (tFinEvTest- tFinEvTrain)), eval, false);
					saveResults(tIni, tFinTrain, (tFinTrain + (tFinEvTest- tFinEvTrain)), eval, false, resultsFile, model);
				}
			}
            
            //If accuracy is lower than the accuracy in the past epoch 
            if (eval.accuracy() < pastAccuracy) {
				log.info("Se ha empeorado en este epoch");
				peorOIgual++;
	        	pastAccuracy=eval.accuracy();

			}
            else if (eval.accuracy() == pastAccuracy){
            	log.info("No se ha mejorado nada en este epoch");
				peorOIgual++;

            }
            else {
            	pastAccuracy=eval.accuracy();
				peorOIgual = 0;

            }
            mnistTest.reset();
            
            
          
		}
		return tIni;
	}


	
	/**
	 * Test model.
	 *
	 * @param outputNum the output num
	 * @param mnistTest the mnist test DataSetIterator
	 * @param model the NN model
	 * @return the evaluation
	 */
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
			
			//Fills the outputValuesInt and the labelsInt
			for (int i = 0; i < output.rows(); i++) {
				//Get an item of the processed batch
				INDArray outputProb = output.getRow(i);
				INDArray labelsProb = labels.getRow(i);
				mostProbableOutput = 0;
				LabelInt = 0;

				//Calculate the int value for the label and the output of the NN
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
	
	/**
	 * Gets the net configuration as string.
	 *
	 * @param model the model
	 * @return the net configuration as string
	 */
	private static String getNetConfigurationAsString(MultiLayerNetwork model){
		
		
		StringBuilder sbConf = new StringBuilder("Configuracion de la red: ");
		List<NeuralNetConfiguration> layers = model.getLayerWiseConfigurations().getConfs();
		for (int i = 0; i < layers.size(); i++) {
			
			sbConf.append("\nLayer "+i+" "+layers.get(i).toJson());
		}
		return sbConf.toString();
	}
	
	/**
	 * Check labels and results with the official labels of the MNIST DataBase.
	 */
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
			
			
			
			//Delete all whitespaces and non visible characters
			refLabelsStr = refLabelsStr.replaceAll("\\s", "");
			
			
			
			String[] singleLabels = refLabelsStr.split(",");
			
			int [] MNISTLabels= new int[singleLabels.length];
			
			for (int i = 0; i < singleLabels.length; i++) {
				MNISTLabels[i] = Integer.parseInt(singleLabels[i]);
			}
			
			log.info("Comparación de labels oficiales y de DL4J...");
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
			
			log.info("Comparación de labels oficiales con resultados de la predicción...");
			
			int errores = 0;
			for (int i = 0; i < MNISTLabels.length; i++) {
				if (MNISTLabels[i] != outputValuesInt[0][i]) {
					errores++;
				}
			}
			
			log.info("Número total de errores en la predicción: "+errores);
		} catch (IOException e) {
			log.error("Error leyendo los labels de "+referenceLabels.getAbsolutePath());
			e.printStackTrace();
		}
		
		
		
	}
	
	/**
	 * Prints the results of an evaluation.
	 *
	 * @param tIni the start time of the training
	 * @param tFinTrain the finish time of the training
	 * @param tFin the finish time of the evaluation
	 * @param eval the evaluation
	 * @param train flag to resolve if the evaluation is a evaluation of the
	 * training set (true) or an evaluation of the test set (false)
	 */
	private static void printResults(long tIni, long tFinTrain, long tFin, Evaluation eval, boolean train) {
	
		log.info("****************Resultados de la evaluación********************");
		log.info(eval.stats());
		log.info("Tiempo total de entrenamiento: " + (tFinTrain - tIni) + " ms");
		log.info("Tiempo total de evaluación: " + (tFin - tFinTrain) + " ms");
		log.info("Tiempo total: " + (tFin - tIni) + " ms");
		long errores = 0;
		
		//Calculate the error counting the differences between the labels and
		//the output of the NN
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
		

		//Prints the output values and the labels as a string of ints without spaces
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


	/**
	 * Wipe output and labels arrays to make another evaluation
	 * without garbage of past evaluations.
	 */
	private static void wipeOutputArrays() {
		for (int i = 0; i < 6; i++) {
        	 Arrays.fill(outputValuesInt[i], 0);
             Arrays.fill(labelsInt[i], 0);
		}
	}

	/**
	 * Save evaluation results, this method do the same as printResults but
	 * writing the results to a file instead than show it in the console.
	 *
	 * @param tIni the start time of the training
	 * @param tFinTrain the finish time of the training
	 * @param tFin the finish time of the evaluation
	 * @param eval the evaluation
	 * @param train flag to resolve if the evaluation is a evaluation of the
	 * training set (true) or an evaluation of the test set (false)
	 * @param file where the results will be saved
	 * @param model the nn model to obtain its configuration
	 */
	private static void saveResults(long tIni, long tFinTrain, long tFin, Evaluation eval, boolean train,
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
			String confStr = getNetConfigurationAsString(model);
			fwr.write(confStr);
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
	

	/**
	 * Enable TrainingUI.
	 * @see https://deeplearning4j.org/visualization.html for more info
	 * @return the stats listener
	 */
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

	/**
	 * Enable TrainingUI from a previously saved stats file.
	 * @see https://deeplearning4j.org/visualization.html for more info
	 * 
	 * @param statsFile file where the stats are saved
	 * @return the stats listener
	 */
	private static void enableUI(File statsFile) {
		//Initialize the user intedrface backend
        UIServer uiServer = UIServer.getInstance();
        
        //Configure where the network information (gradients, activations, score vs. time etc) is to be stored
        //Then add the StatsListener to collect this information from the network, as it trains
        //Alternative: new FileStatsStorage(File) - see UIStorageExample
        StatsStorage statsStorage = new FileStatsStorage(statsFile);
        
        
        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
        uiServer.attach(statsStorage);
        
	}

	/**
	 * Save stats.
	 *
	 * @param stFile the stats file
	 * @return the stats listener to be attached to the model an monitor the
	 * training process 
	 */
	private static StatsListener saveStats(File stFile)
	{
		try {
			
			if (!stFile.exists()) {
				Files.createDirectories(stFile.getParentFile().toPath());
				
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		StatsStorage statsStorage = new FileStatsStorage(stFile);
		StatsListener statListener = new StatsListener(statsStorage);
		return statListener;
	}
	
	/**
	 * Save RNA.
	 *
	 * @param model the nn model
	 * @param modelFile the file where the model is going to be saved
	 * @throws IOException Signals that an I/O exception has occurred.
	 */
	private static void saveRNA(MultiLayerNetwork model, File modelFile) throws IOException {

		boolean saveUpdater = true; // Updater: i.e., the state for Momentum,
									// RMSProp, Adagrad etc. Save this if you
									// want to train your network more in the
									// future

		ModelSerializer.writeModel(model, modelFile, saveUpdater);
	}

	/**
	 * Load RNA.
	 *
	 * @param model the nn model
	 * @param modelFile the file where the model is going to be loaded
	 * @throws IOException Signals that an I/O exception has occurred.
	 */
	private static MultiLayerNetwork loadRNA(File modelFile) throws IOException {

		MultiLayerNetwork restored = ModelSerializer.restoreMultiLayerNetwork(modelFile);
		return restored;
	}

}
