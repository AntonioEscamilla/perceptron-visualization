import { useState, useEffect } from 'react';
import DecisionBoundaryVisualization from './DecisionBoundaryVisualization';

// Definir interfaces para mejorar el tipado
interface TrainingData {
  inputs: [number, number];
  output: number;
}

const PerceptronNetworkAnimation = () => {
  // Datos de entrenamiento para compuerta AND
  const trainingData: TrainingData[] = [
    { inputs: [1, 1], output: 1 },
    { inputs: [1, 0], output: 0 },
    { inputs: [0, 1], output: 0 },
    { inputs: [0, 0], output: 0 }
  ];
  
  // Estados con tipos específicos
  const [weights, setWeights] = useState<[number, number]>([1, 1]);
  const [bias, setBias] = useState<number>(-3);
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [epoch, setEpoch] = useState<number>(0);
  const [learningRate, setLearningRate] = useState<number>(1);
  const [currentSample, setCurrentSample] = useState<number>(0);
  const [prediction, setPrediction] = useState<number | null>(null);
  const [weightedSum, setWeightedSum] = useState<number>(0);
  const [logs, setLogs] = useState<string[]>([]);
  const [isComplete, setIsComplete] = useState<boolean>(false);
  const [isConverged, setIsConverged] = useState<boolean>(false);
  const [phase, setPhase] = useState<'feedforward' | 'backprop'>('feedforward');
  const [currentPredictions, setCurrentPredictions] = useState<(number | null)[]>(new Array(trainingData.length).fill(null));
  
  // Función para predecir la salida del perceptrón
  const predict = (inputs: [number, number], w: [number, number], b: number): number => {
    const sum = inputs[0] * w[0] + inputs[1] * w[1] + b;
    return sum >= 0 ? 1 : 0;
  };
  
  // Calcular la suma ponderada
  const calculateSum = (inputs: [number, number], w: [number, number], b: number): number => {
    return inputs[0] * w[0] + inputs[1] * w[1] + b;
  };
  
  // Calcular el error total
  const calculateTotalError = (w: [number, number], b: number): number => {
    let errorSum = 0;
    for (let i = 0; i < trainingData.length; i++) {
      const sample = trainingData[i];
      const pred = predict(sample.inputs, w, b);
      errorSum += Math.abs(sample.output - pred);
    }
    return errorSum;
  };

  // Función para entrenar al perceptrón
  const trainStep = () => {
    // Si ya se completó todo el proceso de entrenamiento incluyendo la época extra, detener
    if (isComplete) {
      setIsRunning(false);
      return;
    }

    // Obtener muestra actual
    const sample = trainingData[currentSample];
    
    if (phase === 'feedforward') {
      // Fase de propagación hacia adelante
      const sum = calculateSum(sample.inputs, weights, bias);
      setWeightedSum(sum);
      
      // Calcular predicción
      const pred = sum >= 0 ? 1 : 0;
      setPrediction(pred);
      
      // Registrar la predicción (solo en feedforward)
      setCurrentPredictions(prevPredictions => {
        const newPredictions = [...prevPredictions];
        newPredictions[currentSample] = pred;
        return newPredictions;
      });
      
      // Cambiar a fase de propagación hacia atrás
      setPhase('backprop');
    } else {
      // Fase de propagación hacia atrás
      const pred = prediction !== null ? prediction : 0;
      
      // Calcular error
      const error = sample.output - pred;
      
      // Si ya convergió, no actualizamos pesos, solo mostramos
      if (!isConverged) {
        // Actualizar pesos y bias si hay error
        if (error !== 0) {
          const newWeights: [number, number] = [
            weights[0] + learningRate * error * sample.inputs[0],
            weights[1] + learningRate * error * sample.inputs[1]
          ];
          const newBias = bias + learningRate * error;
          
          setWeights(newWeights);
          setBias(newBias);
          
          // Registrar la actualización
          setLogs(prevLogs => [
            ...prevLogs,
            `Época ${epoch}, Muestra [${sample.inputs}] → ${sample.output}, Predicción: ${pred}, Error: ${error}, Pesos: [${newWeights.map(w => w.toFixed(2))}], Bias: ${newBias.toFixed(2)}`
          ].slice(-10));
        } else {
          // Registrar sin actualización
          setLogs(prevLogs => [
            ...prevLogs,
            `Época ${epoch}, Muestra [${sample.inputs}] → ${sample.output}, Predicción: ${pred}, Error: ${error}, Pesos: [${weights.map(w => w.toFixed(2))}], Bias: ${bias.toFixed(2)}`
          ].slice(-10));
        }
      } else {
        // Si ya convergió, solo registramos observación sin actualizar pesos
        setLogs(prevLogs => [
          ...prevLogs,
          `Época extra ${epoch}, Muestra [${sample.inputs}] → ${sample.output}, Predicción: ${pred}, Error: ${error} (Pesos ya finales)`
        ].slice(-10));
      }
      
      // Avanzar a la siguiente muestra
      let nextSample = (currentSample + 1) % trainingData.length;
      setCurrentSample(nextSample);
      
      // Si completamos una época, actualizar contador y verificar si hemos convergido
      if (nextSample === 0) {
        setEpoch(epoch + 1);
        const newTotalError = calculateTotalError(weights, bias);
        
        // Si no había convergido antes y ahora el error es 0, marcar como convergido
        if (!isConverged && newTotalError === 0) {
          setIsConverged(true);
          setLogs(prevLogs => [
            ...prevLogs,
            `¡Convergencia lograda! Ejecutando una época adicional de demostración...`
          ]);
        }
        
        // Si ya convergió y completamos una época extra, entonces hemos terminado
        if (isConverged) {
          setIsComplete(true);
          setLogs(prevLogs => [
            ...prevLogs,
            `Entrenamiento completado. El perceptrón ha aprendido la función AND.`
          ]);
        }
      }
      
      // Volver a fase de propagación hacia adelante
      setPhase('feedforward');
      
      // Reiniciar la predicción para la nueva muestra
      setPrediction(null);
      setWeightedSum(0);
    }
  };
  
  // Función para generar pesos aleatorios
  const randomizeWeights = () => {
    // Generar valores aleatorios entre -3 y 3
    const randomW1 = Math.random() * 6 - 3; // -3 a 3
    const randomW2 = Math.random() * 6 - 3; // -3 a 3
    const randomBias = Math.random() * 6 - 3; // -3 a 3
    
    // Actualizar los estados
    setWeights([randomW1, randomW2]);
    setBias(randomBias);
    
    // Resetear algunas variables para evitar inconsistencias
    setPrediction(null);
    setWeightedSum(0);
    setCurrentPredictions(new Array(trainingData.length).fill(null));
    
    // Agregar un log
    setLogs(prevLogs => [
      ...prevLogs,
      `Pesos generados aleatoriamente: [${randomW1.toFixed(2)}, ${randomW2.toFixed(2)}], Bias: ${randomBias.toFixed(2)}`
    ].slice(-10));
  };
  
  // Efecto para actualizar todas las predicciones cuando se completa el entrenamiento
  useEffect(() => {
    if (isComplete) {
      const allPredictions = trainingData.map(sample => 
        predict(sample.inputs, weights, bias)
      );
      setCurrentPredictions(allPredictions);
    }
  }, [isComplete, weights, bias]);
  
  // Efecto para la animación automática
  useEffect(() => {
    let intervalId: number | undefined;
    
    if (isRunning) {
      intervalId = window.setInterval(trainStep, 1000);
    }
    
    return () => {
      if (intervalId !== undefined) {
        clearInterval(intervalId);
      }
    };
  }, [isRunning, currentSample, weights, bias, epoch, learningRate, phase]);
  
  // Reiniciar el entrenamiento
  const resetTraining = () => {
    setWeights([1, 1]);
    setBias(-3);
    setEpoch(0);
    setCurrentSample(0);
    setPrediction(null);
    setWeightedSum(0);
    setLogs([]);
    setIsComplete(false);
    setIsConverged(false);
    setIsRunning(false);
    setPhase('feedforward');
    setCurrentPredictions(new Array(trainingData.length).fill(null));
  };
  
  // Función para renderizar el perceptrón
  const renderPerceptron = () => {
    const sample = trainingData[currentSample];
    const x1 = sample.inputs[0];
    const x2 = sample.inputs[1];
    const w1 = weights[0];
    const w2 = weights[1];
    const w0 = bias;
    
    // Determinar si mostrar el punto rojo en la función escalón
    // Solo se muestra cuando hay una suma ponderada válida (en fase backprop, después del feedforward)
    const showRedDot = phase === 'backprop' && prediction !== null;
    
    return (
      <svg width="500" height="280" viewBox="0 0 500 280">
        {/* Definir flechas para las conexiones */}
        <defs>
          <marker id="arrowhead" markerWidth="10" markerHeight="7" 
                refX="0" refY="3.5" orient="auto">
            <polygon points="0 0, 10 3.5, 0 7" fill="#000" />
          </marker>
        </defs>
        
        {/* Entradas */}
        <g className="inputs">
          {/* X1 */}
          <circle cx="50" cy="70" r="20" fill="#E3F2FD" stroke="#2196F3" strokeWidth="2" />
          <text x="50" y="75" fontSize="14" textAnchor="middle">x₁={x1}</text>
          
          {/* X2 */}
          <circle cx="50" cy="160" r="20" fill="#E3F2FD" stroke="#2196F3" strokeWidth="2" />
          <text x="50" y="165" fontSize="14" textAnchor="middle">x₂={x2}</text>
          
          {/* Bias */}
          <circle cx="50" cy="250" r="20" fill="#FCE4EC" stroke="#E91E63" strokeWidth="2" />
          <text x="50" y="255" fontSize="14" textAnchor="middle">1</text>
        </g>
        
        {/* Sumador */}
        <g className="summation">
          <circle cx="240" cy="160" r="30" fill="#FFF3E0" stroke="#FF9800" strokeWidth="2" />
          <text x="240" y="170" fontSize="24" textAnchor="middle">Σ</text>
          
          {/* Valor de la suma */}
          <text x="240" y="200" fontSize="12" textAnchor="middle">
            Suma: {weightedSum.toFixed(2)}
          </text>
        </g>
        
        {/* Función de activación */}
        <g className="activation">
          <circle cx="360" cy="160" r="30" fill="#F1F8E9" stroke="#8BC34A" strokeWidth="2" />
          
          {/* Función escalón mejorada */}
          <svg x="330" y="130" width="60" height="60" viewBox="0 0 60 60" overflow="visible">
            {/* Eje X */}
            <line x1="0" y1="50" x2="60" y2="50" stroke="#777" strokeWidth="0.5" />
            
            {/* Eje Y */}
            <line x1="20" y1="0" x2="20" y2="50" stroke="#777" strokeWidth="0.5" />
            
            {/* Etiqueta de la función */}
            <text x="30" y="5" fontSize="10" textAnchor="middle">f(x)</text>
            
            {/* Función escalón */}
            <line x1="0" y1="40" x2="20" y2="40" stroke="black" strokeWidth="2" /> {/* Parte horizontal para x<0 (valor 0) */}
            <line x1="20" y1="40" x2="20" y2="20" stroke="black" strokeWidth="2" /> {/* Parte vertical */}
            <line x1="20" y1="20" x2="60" y2="20" stroke="black" strokeWidth="2" /> {/* Parte horizontal para x>=0 (valor 1) */}
            
            {/* Valores de la función */}
            <text x="10" y="38" fontSize="8" textAnchor="middle" fill="#666">0</text>
            <text x="40" y="18" fontSize="8" textAnchor="middle" fill="#666">1</text>
            
            {/* Punto que muestra el valor actual en la función - solo visible después de un feedforward */}
            {showRedDot && (
              <circle 
                cx={weightedSum < 0 ? 10 : 40} 
                cy={weightedSum < 0 ? 40 : 20} 
                r="3" 
                fill="red" 
              />
            )}
          </svg>
        </g>
        
        {/* Salida */}
        <g className="output">
          <circle cx="450" cy="160" r="20" fill="#FFEBEE" stroke="#F44336" strokeWidth={prediction === 1 ? "4" : "2"} />
          <text x="450" y="165" fontSize="16" textAnchor="middle">{prediction === null ? '?' : prediction}</text>
        </g>
        
        {/* Conexiones */}
        <g className="connections">
          {/* X1 a Sumador - sin flecha */}
          <line x1="70" y1="70" x2="212" y2="150" stroke={w1 >= 0 ? "#4CAF50" : "#F44336"} 
                strokeWidth={Math.abs(w1) * 1.5 + 1} />
          <text x="130" y="100" fontSize="12" fontWeight="bold">{w1.toFixed(2)}</text>
          
          {/* X2 a Sumador - sin flecha */}
          <line x1="70" y1="160" x2="210" y2="160" stroke={w2 >= 0 ? "#4CAF50" : "#F44336"} 
                strokeWidth={Math.abs(w2) * 1.5 + 1} />
          <text x="130" y="150" fontSize="12" fontWeight="bold">{w2.toFixed(2)}</text>
          
          {/* Bias a Sumador - sin flecha */}
          <line x1="70" y1="250" x2="212" y2="170" stroke={w0 >= 0 ? "#4CAF50" : "#F44336"} 
                strokeWidth={Math.abs(w0) * 1.5 + 1} />
          <text x="130" y="220" fontSize="12" fontWeight="bold">{w0.toFixed(2)}</text>
          
          {/* Sumador a Función de activación - MODIFICADO para que la flecha no esté dentro del círculo */}
          <line x1="270" y1="160" x2="310" y2="160" stroke="#000" 
                strokeWidth="2" markerEnd="url(#arrowhead)" />
          
          {/* Función de activación a Salida - MODIFICADO para que la flecha no esté dentro del círculo */}
          <line x1="390" y1="160" x2="410" y2="160" stroke="#000" 
                strokeWidth="2" markerEnd="url(#arrowhead)" />
        </g>
        
        {/* Target y predicción - solo mostrar después del feedforward */}
        {(phase === 'backprop' || isComplete) && (
          <g className="target">
            <rect x="370" y="80" width="120" height="40" fill="#F5F5F5" stroke="#9E9E9E" strokeWidth="1" rx="5" />
            <text x="430" y="100" fontSize="12" textAnchor="middle">
              Target: {sample.output}
            </text>
            <text x="430" y="115" fontSize="12" textAnchor="middle">
              Error: {sample.output - (prediction ?? 0)}
            </text>
          </g>
        )}
        
        {/* Fase actual */}
        <rect x="240" y="30" width="160" height="30" fill={phase === 'feedforward' ? "#E3F2FD" : "#FFF3E0"} stroke="#9E9E9E" strokeWidth="1" rx="5" />
        <text x="320" y="50" fontSize="14" fontWeight="bold" textAnchor="middle">
          Fase: {phase === 'feedforward' ? "Forward" : "Backprop"}
        </text>
      </svg>
    );
  };
  
  return (
    <div className="flex flex-col items-center max-w-4xl mx-auto p-4 bg-gray-50 rounded-lg shadow">
      <h2 className="text-2xl font-bold mb-6">Animación de Perceptrón - Compuerta AND</h2>
      
      {/* Contenedor principal con dos paneles lado a lado */}
      <div className="flex flex-col md:flex-row w-full gap-4 mb-4">
        {/* Panel izquierdo: Perceptrón */}
        <div className="w-full md:w-1/2 bg-white rounded-lg shadow p-4">
          <h3 className="font-bold mb-4 text-center">Representación del Modelo</h3>
          <div className="flex justify-center mb-4">
            {renderPerceptron()}
          </div>
        </div>
        
        {/* Panel derecho: Controles */}
        <div className="w-full md:w-1/2 bg-white rounded-lg shadow p-4">
          <h3 className="font-bold mb-4">Controles</h3>
          <div className="flex justify-center gap-2 mb-4">
            <button 
              onClick={() => setIsRunning(!isRunning)} 
              className={`px-6 py-3 rounded ${isComplete ? 'bg-gray-300' : (isRunning ? 'bg-red-500 hover:bg-red-600' : 'bg-green-500 hover:bg-green-600')} text-white`}
              disabled={isComplete}
            >
              {isRunning ? 'Pausar' : 'Auto'}
            </button>
            <button 
              onClick={trainStep} 
              className={`px-6 py-3 rounded text-white`}
              style={{backgroundColor: phase === 'feedforward' ? '#3b82f6' : '#FF9800'}}
              disabled={isRunning || isComplete}
            >
              {phase === 'feedforward' ? 'Forward ▶' : 'Backprop ◀'}
            </button>
            <button 
              onClick={resetTraining} 
              className="px-6 py-3 rounded bg-blue-500 hover:bg-blue-600 text-white"
            >
              Reiniciar
            </button>
          </div>
          
          <div className="flex items-center gap-4 mb-6">
            <label className="whitespace-nowrap">Tasa de aprendizaje:</label>
            <input 
              type="range" 
              min="0.01" 
              max="1" 
              step="0.01" 
              value={learningRate} 
              onChange={e => setLearningRate(parseFloat(e.target.value))}
              className="w-full"
            />
            <span className="font-mono w-12 text-right">{learningRate.toFixed(2)}</span>
          </div>
          
          <div className="mt-4">
            <h4 className="font-bold mb-2">Leyenda:</h4>
            <ul className="text-sm">
              <li className="flex items-center mb-1"><span className="inline-block w-3 h-3 bg-green-500 rounded-full mr-1"></span> Peso positivo</li>
              <li className="flex items-center mb-1"><span className="inline-block w-3 h-3 bg-red-500 rounded-full mr-1"></span> Peso negativo</li>
              <li className="flex items-center mb-1"><span className="inline-block w-3 h-3 bg-yellow-500 rounded-full mr-1"></span> Sumador</li>
              <li className="flex items-center mb-1"><span className="inline-block w-3 h-3 bg-green-200 rounded-full mr-1"></span> Función de activación (escalón)</li>
            </ul>

            <div className="mt-4 flex justify-center">
              <button 
                onClick={randomizeWeights} 
                className="px-6 py-3 rounded bg-purple-500 hover:bg-purple-600 text-white"
              >
                Pesos Random
              </button>
            </div>

          </div>
        </div>
      </div>
      
      <div className="w-full bg-white rounded-lg shadow p-4 mb-4">
        <h3 className="font-bold mb-2">Estado del Entrenamiento</h3>
        <div className="flex flex-col space-y-2">
          <div className="flex items-center">
            <div className="w-32">Estado:</div>
            <div 
              className="font-medium" 
              style={{ 
                color: isConverged 
                  ? (isComplete ? '#16a34a' : '#2563eb') 
                  : '#FF9800'
              }}
            >
              {isComplete 
                ? '✓ Entrenamiento completado' 
                : (isConverged 
                  ? '✓ Convergencia lograda - Ejecutando época adicional' 
                  : '⟳ Entrenando...')}
            </div>
          </div>
        </div>
      </div>
      
      <div className="flex flex-wrap justify-center gap-4 w-full">
        <div className="bg-white rounded-lg shadow p-4 flex-1">
          <h3 className="font-bold mb-2">Estado Actual</h3>
          <div className="grid grid-cols-2 gap-2 mb-4">
            <div>Fase actual:</div>
            <div className={`font-mono ${phase === 'feedforward' ? 'text-blue-500' : 'text-red-500'}`}>
              {phase === 'feedforward' ? 'Forward-propagation' : 'Back-propagation'}
            </div>
            
            <div>Época:</div>
            <div className="font-mono">{epoch}</div>
            
            <div>Pesos:</div>
            <div className="font-mono">[{weights.map(w => w.toFixed(3)).join(', ')}]</div>
            
            <div>Bias:</div>
            <div className="font-mono">{bias.toFixed(3)}</div>
            
            <div>Tasa de aprendizaje:</div>
            <div className="font-mono">{learningRate}</div>
            
            <div>Error:</div>
            <div className="font-mono">
              {phase === 'backprop' || prediction !== null ? 
                (trainingData[currentSample].output - (prediction ?? 0)) : '...'}
            </div>
            
            <div>Muestra actual:</div>
            <div className="font-mono">[{trainingData[currentSample].inputs.join(', ')}] → {trainingData[currentSample].output}</div>
            
            <div>Salida del sumador:</div>
            <div className="font-mono">{weightedSum.toFixed(3)}</div>
            
            <div>Predicción:</div>
            <div className={`font-mono ${
              prediction === null 
                ? 'text-gray-500' 
                : (prediction === trainingData[currentSample].output 
                    ? 'text-green-500' 
                    : 'text-red-500')
            }`}>
              {prediction === null ? '...' : prediction}
            </div>
          </div>
        </div>
        
        <div className="bg-white rounded-lg shadow p-4 flex-1">
          <h3 className="font-bold mb-2">Tabla de Verdad AND</h3>
          <table className="w-full table-auto border-collapse mb-4">
            <thead>
              <tr className="bg-gray-100">
                <th className="border p-1">X₁</th>
                <th className="border p-1">X₂</th>
                <th className="border p-1">Esperado</th>
                <th className="border p-1">Predicción</th>
              </tr>
            </thead>
            <tbody>
              {trainingData.map((sample, idx) => (
                <tr key={idx} className={idx === currentSample ? 'bg-yellow-100' : ''}>
                  <td className="border p-1 text-center">{sample.inputs[0]}</td>
                  <td className="border p-1 text-center">{sample.inputs[1]}</td>
                  <td className="border p-1 text-center">{sample.output}</td>
                  <td className={`border p-1 text-center ${
                    currentPredictions[idx] === null 
                      ? '' 
                      : currentPredictions[idx] === sample.output 
                        ? 'text-green-500 font-bold' 
                        : 'text-red-500 font-bold'
                  }`}>
                    {currentPredictions[idx] !== null ? currentPredictions[idx] : ''}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      
      {/* Visualización del Espacio de Decisión */}
      <DecisionBoundaryVisualization
        weights={weights}
        bias={bias}
        trainingData={trainingData}
        currentSample={currentSample}
        isRunning={isRunning}
      />
      
      <div className="w-full bg-white rounded-lg shadow p-4 mb-4">
        <h3 className="font-bold mb-2">Registro de Actualización de Pesos</h3>
        <div className="h-32 overflow-y-auto bg-gray-100 p-2 rounded text-xs font-mono">
          {logs.length === 0 ? (
            <p>Inicie el entrenamiento para ver el registro...</p>
          ) : (
            logs.map((log, idx) => <p key={idx}>{log}</p>)
          )}
        </div>
      </div>
    </div>
  );
};

export default PerceptronNetworkAnimation;