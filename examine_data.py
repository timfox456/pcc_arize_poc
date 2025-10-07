
def generate_analytics(df):
        """Generate basic analytics from the data"""
        
        # Basic metrics
        accuracy = (df['prediction_score'].round() == df['actual_label']).mean()
        avg_prediction = df['prediction_score'].mean()
        fraud_rate = df['actual_label'].mean()
        
        print("\n" + "="*50)
        print("DEMO ANALYTICS SUMMARY")
        print("="*50)
        print(f"Total Predictions: {len(df):,}")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Average Prediction Score: {avg_prediction:.2%}")
        print(f"Actual Fraud Rate: {fraud_rate:.2%}")
        print(f"Data Date Range: {df['prediction_timestamp'].min()} to {df['prediction_timestamp'].max()}")
        
        # Feature insights
        print("\nFEATURE DISTRIBUTIONS:")
        print(f"Average Transaction Amount: ${df['transaction_amount'].mean():.2f}")
        print(f"Average User History: {df['user_history_days'].mean():.0f} days")
        print(f"Device Types: {df['device_type'].value_counts().to_dict()}")
        print(f"Average Location Risk Score: {df['location_risk'].mean():.3f}")
        
        return {
            'accuracy': accuracy,
            'avg_prediction': avg_prediction,
            'fraud_rate': fraud_rate,
            'total_predictions': len(df)
        }