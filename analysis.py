import numpy as np
from typing import Dict, Any

class AnalysisGenerator:
    """Classe para gerar insights automáticos em linguagem natural"""
    
    def __init__(self):
        pass
    
    def generate_model_insights(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Gera insights para cada modelo"""
        insights = {}
        
        for model_name, metrics in results.items():
            if 'error' in metrics:
                insights[model_name] = f"❌ Erro ao treinar o modelo {model_name}: {metrics['error']}"
                continue
            
            problem_type = metrics.get('problem_type', 'classification')
            insight_parts = []
            
            if problem_type == 'classification':
                # Análise da acurácia
                accuracy = metrics.get('accuracy', 0)
                if accuracy >= 0.9:
                    insight_parts.append(f"✅ Excelente acurácia ({accuracy:.1%})")
                elif accuracy >= 0.8:
                    insight_parts.append(f"✅ Boa acurácia ({accuracy:.1%})")
                elif accuracy >= 0.7:
                    insight_parts.append(f"⚠️ Acurácia moderada ({accuracy:.1%})")
                else:
                    insight_parts.append(f"❌ Acurácia baixa ({accuracy:.1%})")
                
                # Análise da precisão
                precision = metrics.get('precision', 0)
                if precision < 0.7:
                    insight_parts.append("⚠️ Precisão baixa - modelo gera muitos falsos positivos")
                elif precision >= 0.9:
                    insight_parts.append("✅ Excelente precisão")
                
                # Análise do recall
                recall = metrics.get('recall', 0)
                if recall < 0.7:
                    insight_parts.append("⚠️ Recall baixo - modelo perde muitos casos positivos")
                elif recall >= 0.9:
                    insight_parts.append("✅ Excelente recall")
                
                # Análise do F1-score
                f1 = metrics.get('f1', 0)
                if f1 >= 0.9:
                    insight_parts.append("✅ F1-score excelente - bom balanceamento")
                elif f1 < 0.7:
                    insight_parts.append("⚠️ F1-score baixo - desbalanceamento entre precisão e recall")
                
                # ROC-AUC se disponível
                roc_auc = metrics.get('roc_auc')
                if roc_auc is not None and not np.isnan(roc_auc):
                    if roc_auc >= 0.9:
                        insight_parts.append(f"✅ ROC-AUC excelente ({roc_auc:.3f})")
                    elif roc_auc < 0.7:
                        insight_parts.append(f"⚠️ ROC-AUC baixo ({roc_auc:.3f})")
                        
            else:  # regression
                # Análise do R² Score
                r2 = metrics.get('r2_score', 0)
                if r2 >= 0.9:
                    insight_parts.append(f"✅ Excelente R² ({r2:.3f}) - modelo explica muito bem a variância")
                elif r2 >= 0.8:
                    insight_parts.append(f"✅ Bom R² ({r2:.3f}) - modelo explica bem a variância")
                elif r2 >= 0.6:
                    insight_parts.append(f"⚠️ R² moderado ({r2:.3f}) - modelo explica parte da variância")
                elif r2 >= 0:
                    insight_parts.append(f"❌ R² baixo ({r2:.3f}) - modelo explica pouco da variância")
                else:
                    insight_parts.append(f"❌ R² negativo ({r2:.3f}) - modelo pior que a média")
                
                # Análise do RMSE
                rmse = metrics.get('rmse', 0)
                y_mean = metrics.get('y_test_mean', 1)
                rmse_relative = rmse / y_mean if y_mean != 0 else float('inf')
                
                if rmse_relative < 0.1:
                    insight_parts.append(f"✅ RMSE baixo ({rmse:.3f}) - predições muito precisas")
                elif rmse_relative < 0.2:
                    insight_parts.append(f"✅ RMSE bom ({rmse:.3f}) - predições precisas")
                elif rmse_relative < 0.5:
                    insight_parts.append(f"⚠️ RMSE moderado ({rmse:.3f}) - predições razoáveis")
                else:
                    insight_parts.append(f"❌ RMSE alto ({rmse:.3f}) - predições imprecisas")
                
                # Análise do MAE
                mae = metrics.get('mae', 0)
                mae_relative = mae / y_mean if y_mean != 0 else float('inf')
                
                if mae_relative < 0.1:
                    insight_parts.append("✅ Erro médio baixo")
                elif mae_relative > 0.3:
                    insight_parts.append("⚠️ Erro médio alto")
            
            insights[model_name] = " | ".join(insight_parts)
        
        return insights
    
    def generate_comparison_insights(self, results: Dict[str, Any]) -> str:
        """Gera insights comparativos entre modelos"""
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not valid_results:
            return "❌ Nenhum modelo foi treinado com sucesso."
        
        if len(valid_results) == 1:
            model_name = list(valid_results.keys())[0]
            return f"✅ Apenas o modelo {model_name} foi treinado com sucesso."
        
        # Detecta tipo de problema
        problem_type = list(valid_results.values())[0].get('problem_type', 'classification')
        
        insights = []
        
        if problem_type == 'classification':
            # Encontra o melhor modelo por F1-score
            best_model = max(valid_results.items(), key=lambda x: x[1].get('f1', 0))
            best_name, best_metrics = best_model
            
            insights.append(f"🏆 Melhor modelo: {best_name} (F1-score: {best_metrics.get('f1', 0):.3f})")
            
            # Compara métricas
            accuracies = {k: v.get('accuracy', 0) for k, v in valid_results.items()}
            highest_accuracy = max(accuracies.items(), key=lambda x: x[1])
            
            if highest_accuracy[0] != best_name:
                insights.append(f"📊 Maior acurácia: {highest_accuracy[0]} ({highest_accuracy[1]:.3f})")
            
            # Análise geral
            avg_accuracy = np.mean(list(accuracies.values()))
            if avg_accuracy >= 0.85:
                insights.append("✅ Todos os modelos apresentaram boa performance geral")
            elif avg_accuracy < 0.7:
                insights.append("⚠️ Performance geral baixa - considere revisar os dados")
                
        else:  # regression
            # Encontra o melhor modelo por R² score
            best_model = max(valid_results.items(), key=lambda x: x[1].get('r2_score', -float('inf')))
            best_name, best_metrics = best_model
            
            insights.append(f"🏆 Melhor modelo: {best_name} (R²: {best_metrics.get('r2_score', 0):.3f})")
            
            # Compara RMSE (menor é melhor)
            rmse_values = {k: v.get('rmse', float('inf')) for k, v in valid_results.items()}
            lowest_rmse = min(rmse_values.items(), key=lambda x: x[1])
            
            if lowest_rmse[0] != best_name:
                insights.append(f"📉 Menor RMSE: {lowest_rmse[0]} ({lowest_rmse[1]:.3f})")
            
            # Análise geral baseada em R²
            r2_values = [v.get('r2_score', 0) for v in valid_results.values()]
            avg_r2 = np.mean(r2_values)
            
            if avg_r2 >= 0.8:
                insights.append("✅ Todos os modelos apresentaram boa capacidade explicativa")
            elif avg_r2 < 0.5:
                insights.append("⚠️ Capacidade explicativa baixa - considere revisar features")
        
        return " | ".join(insights)
    
    def generate_recommendations(self, results: Dict[str, Any], data_info: Dict[str, Any]) -> list:
        """Gera recomendações de melhoria"""
        recommendations = []
        
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not valid_results:
            return ["🔧 Revisar qualidade dos dados - nenhum modelo foi treinado com sucesso"]
        
        # Análise do dataset
        rows = data_info.get('rows', 0)
        if rows < 1000:
            recommendations.append(f"📈 Dataset pequeno ({rows} linhas) - coletar mais dados pode melhorar a performance")
        
        features = data_info.get('features', 0)
        if features < 3:
            recommendations.append("🔧 Poucas features - considere feature engineering para criar novas variáveis")
        
        # Análise da performance
        avg_f1 = np.mean([v.get('f1', 0) for v in valid_results.values()])
        
        if avg_f1 < 0.8:
            recommendations.extend([
                "🎯 Otimizar hiperparâmetros com GridSearchCV ou RandomizedSearchCV",
                "🔄 Experimentar técnicas de cross-validation",
                "🧪 Testar outros algoritmos (XGBoost, LightGBM, Neural Networks)"
            ])
        
        if avg_f1 < 0.7:
            recommendations.extend([
                "🔍 Análise exploratória dos dados mais detalhada",
                "🧹 Verificar outliers e ruídos nos dados",
                "⚖️ Verificar balanceamento das classes"
            ])
        
        # Recomendações específicas por tipo de problema
        n_classes = data_info.get('target_classes', 2)
        if n_classes > 2:
            recommendations.append("📊 Para problemas multiclasse, considere estratégias one-vs-rest ou ensemble methods")
        
        if not recommendations:
            recommendations.append("✅ Modelos com boa performance - considere deploy em produção")
        
        return recommendations
    
    def generate_full_analysis(self, pipeline_result: Dict[str, Any]) -> Dict[str, Any]:
        """Gera análise completa dos resultados"""
        if not pipeline_result.get('success'):
            return {
                'success': False,
                'error': pipeline_result.get('error', 'Erro desconhecido'),
                'analysis': 'Não foi possível gerar análise devido a erro no pipeline.'
            }
        
        results = pipeline_result.get('results', {})
        data_info = pipeline_result.get('data_info', {})
        
        # Gera insights
        model_insights = self.generate_model_insights(results)
        comparison = self.generate_comparison_insights(results)
        recommendations = self.generate_recommendations(results, data_info)
        
        # Resumo executivo
        valid_models = len([r for r in results.values() if 'error' not in r])
        total_models = len(results)
        
        executive_summary = []
        executive_summary.append(f"📋 {valid_models}/{total_models} modelos treinados com sucesso")
        executive_summary.append(f"📊 Dataset: {data_info.get('rows', 0)} linhas, {data_info.get('features', 0)} features")
        executive_summary.append(f"🎯 {data_info.get('target_classes', 0)} classes para predição")
        
        return {
            'success': True,
            'executive_summary': executive_summary,
            'model_insights': model_insights,
            'comparison': comparison,
            'recommendations': recommendations,
            'data_info': data_info,
            'timestamp': pipeline_result.get('timestamp')
        }