import { useEffect, useRef } from 'react';

interface DecisionBoundaryProps {
  weights: [number, number];
  bias: number;
  trainingData: Array<{ inputs: [number, number]; output: number }>;
  currentSample: number;
  isRunning: boolean;
}

const DecisionBoundaryVisualization: React.FC<DecisionBoundaryProps> = ({
  weights,
  bias,
  trainingData,
  currentSample,
  isRunning
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  // Función para dibujar el espacio de decisión
  const drawDecisionBoundary = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    
    // Limpiar el canvas
    ctx.clearRect(0, 0, width, height);
    
    // Configuración de escala y transformación
    const scale = width / 8; // Escala para rango -4 a 4 (total 8 unidades)
    const xOffset = width / 2;
    const yOffset = height / 2;
    
    // Dibujar ejes
    ctx.strokeStyle = '#aaa';
    ctx.lineWidth = 1;
    
    // Eje X
    ctx.beginPath();
    ctx.moveTo(0, yOffset);
    ctx.lineTo(width, yOffset);
    ctx.stroke();
    
    // Eje Y
    ctx.beginPath();
    ctx.moveTo(xOffset, 0);
    ctx.lineTo(xOffset, height);
    ctx.stroke();
    
    // Etiquetas de ejes
    ctx.fillStyle = '#666';
    ctx.font = '12px Arial';
    ctx.fillText('X₁', width - 15, yOffset - 5);
    ctx.fillText('X₂', xOffset + 5, 15);
    
    // Transformar coordenadas lógicas (0,0->1,1) a coordenadas del canvas
    const transformX = (x: number) => xOffset + x * scale;
    const transformY = (y: number) => yOffset - y * scale; // Invertido porque en canvas Y crece hacia abajo
    
    // Dibujar la cuadrícula
    ctx.strokeStyle = '#eee';
    ctx.lineWidth = 1;
    
    // Líneas verticales
    for (let x = -4; x <= 4; x += 1) {
      ctx.beginPath();
      ctx.moveTo(transformX(x), 0);
      ctx.lineTo(transformX(x), height);
      ctx.stroke();
      
      // Etiquetas para puntos enteros
      if (Number.isInteger(x)) {
        ctx.fillText(x.toString(), transformX(x) - 5, yOffset + 15);
      }
    }
    
    // Líneas horizontales
    for (let y = -4; y <= 4; y += 1) {
      ctx.beginPath();
      ctx.moveTo(0, transformY(y));
      ctx.lineTo(width, transformY(y));
      ctx.stroke();
      
      // Etiquetas para puntos enteros
      if (Number.isInteger(y)) {
        ctx.fillText(y.toString(), xOffset - 15, transformY(y) + 5);
      }
    }
    
    // Dibujar línea de decisión
    // Ecuación de la línea: w1*x1 + w2*x2 + bias = 0
    // Despejando x2: x2 = (-w1*x1 - bias) / w2
    ctx.strokeStyle = '#2196F3';
    ctx.lineWidth = 2;
    
    // Si w2 es aproximadamente 0, la línea es vertical
    if (Math.abs(weights[1]) < 0.001) {
      const x = -bias / weights[0];
      ctx.beginPath();
      ctx.moveTo(transformX(x), 0);
      ctx.lineTo(transformX(x), height);
      ctx.stroke();
    } else {
      // Calcular dos puntos para dibujar la línea
      const calcY = (x: number) => (-weights[0] * x - bias) / weights[1];
      
      // Puntos para la línea
      const x1 = -4;
      const y1 = calcY(x1);
      const x2 = 4;
      const y2 = calcY(x2);
      
      ctx.beginPath();
      ctx.moveTo(transformX(x1), transformY(y1));
      ctx.lineTo(transformX(x2), transformY(y2));
      ctx.stroke();
    }
    
    // Dibujar los puntos de entrenamiento
    trainingData.forEach((point, index) => {
      const [x, y] = point.inputs;
      const output = point.output;
      
      // Dibujar punto
      ctx.beginPath();
      ctx.arc(transformX(x), transformY(y), 8, 0, Math.PI * 2);
      
      // Destacar muestra actual si el entrenamiento está en curso
      if (index === currentSample && isRunning) {
        ctx.fillStyle = '#FFC107'; // Amarillo para la muestra actual
        ctx.strokeStyle = '#FF9800';
        ctx.lineWidth = 2;
      } else {
        // Color según la clase (etiqueta)
        ctx.fillStyle = output === 1 ? '#4CAF50' : '#F44336';
        ctx.strokeStyle = '#000';
        ctx.lineWidth = 1;
      }
      
      ctx.fill();
      ctx.stroke();
      
      // Etiqueta con coordenadas
      ctx.fillStyle = '#000';
      ctx.font = '10px Arial';
      ctx.fillText(`(${x},${y})`, transformX(x) + 10, transformY(y) - 10);
    });
    
    // Leyenda
    ctx.font = '12px Arial';
    ctx.fillStyle = '#000';
    ctx.fillText('Leyenda:', 10, height - 60);
    
    // Punto clase 1
    ctx.beginPath();
    ctx.arc(20, height - 40, 6, 0, Math.PI * 2);
    ctx.fillStyle = '#4CAF50';
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = '#000';
    ctx.fillText('Clase 1 (AND = 1)', 30, height - 36);
    
    // Punto clase 0
    ctx.beginPath();
    ctx.arc(20, height - 20, 6, 0, Math.PI * 2);
    ctx.fillStyle = '#F44336';
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = '#000';
    ctx.fillText('Clase 0 (AND = 0)', 30, height - 16);
    
    // Ecuación de la línea de decisión
    const w1 = weights[0].toFixed(2);
    const w2 = weights[1].toFixed(2);
    const b = bias.toFixed(2);
    const sign = bias >= 0 ? '+' : '';
    
    ctx.font = 'bold 14px Arial';
    ctx.fillStyle = '#2196F3';
    ctx.fillText(`Línea de decisión: ${w1}x₁ + ${w2}x₂ ${sign}${b} = 0`, width / 2 - 150, 20);
  };
  
  // Efecto para dibujar cuando cambien los pesos o el bias
  useEffect(() => {
    drawDecisionBoundary();
  }, [weights, bias, currentSample, isRunning]);
  
  // Efecto para redimensionar el canvas
  useEffect(() => {
    const handleResize = () => {
      if (canvasRef.current) {
        const container = canvasRef.current.parentElement;
        if (container) {
          // Determinar el tamaño para un canvas cuadrado
          const size = Math.min(container.clientWidth, 500); // Limitamos a 500px máximo
          
          // Establecer ancho y alto iguales para un canvas cuadrado
          canvasRef.current.width = size;
          canvasRef.current.height = size;
          
          drawDecisionBoundary();
        }
      }
    };
    
    // Configurar tamaño inicial
    handleResize();
    
    // Escuchar cambios de tamaño
    window.addEventListener('resize', handleResize);
    
    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, []);
  
  return (
    <div className="w-full bg-white rounded-lg shadow p-4 mb-4">
      <h3 className="font-bold mb-4 text-center">Visualización del Espacio de Decisión</h3>
      <div className="flex justify-center">
        <div className="w-full max-w-lg"> {/* Contenedor para limitar el ancho máximo */}
          <canvas 
            ref={canvasRef} 
            className="border border-gray-200 rounded mx-auto" /* mx-auto centra el canvas */
            width={500} 
            height={500}
          />
        </div>
      </div>
    </div>
  );
};

export default DecisionBoundaryVisualization;