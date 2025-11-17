"""
Sample Data Generator
Creates a small sample dataset for testing if you don't have the full This American Life dataset yet
"""

import pandas as pd
import numpy as np

def create_sample_dataset():
    """
    Generate sample podcast transcript data for testing
    """
    
    sample_episodes = [
        {
            'episode_id': 'ep001',
            'episode_title': 'Family Stories',
            'segments': [
                {
                    'speaker': 'Ira Glass',
                    'text': 'Today on the show, we bring you stories about families. About the bonds that hold us together and the tensions that pull us apart. We have three stories about parents and children, and the complicated love between them.',
                    'start_time': 0.0,
                    'end_time': 15.0
                },
                {
                    'speaker': 'Narrator',
                    'text': 'My father was a quiet man. He worked long hours at the factory and came home tired. But every Sunday, without fail, he would take me fishing. Those were the moments I felt closest to him, sitting by the lake in silence.',
                    'start_time': 15.0,
                    'end_time': 35.0
                },
                {
                    'speaker': 'Ira Glass',
                    'text': 'That story reminds us how the smallest rituals can mean everything in a family. How showing up, even in silence, can be a form of love.',
                    'start_time': 35.0,
                    'end_time': 48.0
                }
            ],
            'date': '2020-01-15'
        },
        {
            'episode_id': 'ep002',
            'episode_title': 'The Journey Home',
            'segments': [
                {
                    'speaker': 'Ira Glass',
                    'text': 'This week, stories about immigration and finding home. About leaving everything behind and starting over in a new place. About the courage it takes to rebuild your life.',
                    'start_time': 0.0,
                    'end_time': 18.0
                },
                {
                    'speaker': 'Maria',
                    'text': 'I came to America with just a suitcase and a dream. I left my family in Mexico, my friends, everything I knew. The first year was so hard. I cried every night. But I kept going, kept working, because I wanted a better future.',
                    'start_time': 18.0,
                    'end_time': 42.0
                },
                {
                    'speaker': 'Maria',
                    'text': 'Now, ten years later, I have my own restaurant. My children are going to college. Sometimes I still cry, but now they are tears of gratitude. This country gave me a second chance.',
                    'start_time': 42.0,
                    'end_time': 62.0
                }
            ],
            'date': '2020-02-20'
        },
        {
            'episode_id': 'ep003',
            'episode_title': 'Acts of Forgiveness',
            'segments': [
                {
                    'speaker': 'Ira Glass',
                    'text': 'Today we explore forgiveness. What does it mean to forgive someone who has hurt you deeply? Is forgiveness always possible? We have stories of people who found a way to let go of anger and move forward.',
                    'start_time': 0.0,
                    'end_time': 20.0
                },
                {
                    'speaker': 'David',
                    'text': 'My brother stole from me. Not just money, but my trust. For five years, I refused to speak to him. I held onto that anger like a shield. But it was poisoning me. Eventually, I realized that forgiveness was not about letting him off the hook. It was about freeing myself.',
                    'start_time': 20.0,
                    'end_time': 48.0
                },
                {
                    'speaker': 'Ira Glass',
                    'text': 'Forgiveness is complicated. It does not mean forgetting or condoning what happened. It means choosing to release the burden of resentment.',
                    'start_time': 48.0,
                    'end_time': 60.0
                }
            ],
            'date': '2020-03-10'
        },
        {
            'episode_id': 'ep004',
            'episode_title': 'Growing Up Different',
            'segments': [
                {
                    'speaker': 'Ira Glass',
                    'text': 'Stories about childhood and feeling different. About the kids who did not quite fit in, and how they found their way. About learning to embrace what makes you unique.',
                    'start_time': 0.0,
                    'end_time': 17.0
                },
                {
                    'speaker': 'Sarah',
                    'text': 'I was the only Asian kid in my school. I tried so hard to fit in. I wanted to be blonde, to have a different name, to eat different food. It took me years to realize that my difference was not a weakness. It was my strength.',
                    'start_time': 17.0,
                    'end_time': 38.0
                },
                {
                    'speaker': 'Narrator',
                    'text': 'Studies show that children who feel different often develop greater empathy and creativity. What seems like a burden in childhood can become a gift in adulthood.',
                    'start_time': 38.0,
                    'end_time': 52.0
                }
            ],
            'date': '2020-04-05'
        },
        {
            'episode_id': 'ep005',
            'episode_title': 'Justice and Mercy',
            'segments': [
                {
                    'speaker': 'Ira Glass',
                    'text': 'This week, we look at the criminal justice system. At stories of crime and punishment, of victims and perpetrators. At the complex question of what justice really means.',
                    'start_time': 0.0,
                    'end_time': 19.0
                },
                {
                    'speaker': 'Judge Williams',
                    'text': 'I have sentenced hundreds of people. Each case is different. I try to balance punishment with rehabilitation. To hold people accountable while also giving them a chance to change. It is not easy. Sometimes I wonder if I got it right.',
                    'start_time': 19.0,
                    'end_time': 42.0
                },
                {
                    'speaker': 'Narrator',
                    'text': 'The United States has the highest incarceration rate in the world. Many advocates argue for criminal justice reform, for focusing more on rehabilitation and less on punishment. But change is slow.',
                    'start_time': 42.0,
                    'end_time': 60.0
                }
            ],
            'date': '2020-05-12'
        }
    ]
    
    # Convert to flat DataFrame format
    rows = []
    for episode in sample_episodes:
        for segment in episode['segments']:
            rows.append({
                'episode_id': episode['episode_id'],
                'episode_title': episode['episode_title'],
                'speaker': segment['speaker'],
                'text': segment['text'],
                'start_time': segment['start_time'],
                'end_time': segment['end_time'],
                'date': episode['date']
            })
    
    df = pd.DataFrame(rows)
    return df

if __name__ == "__main__":
    print("Creating sample This American Life dataset...")
    df = create_sample_dataset()
    
    # Save to CSV
    df.to_csv("this_american_life_transcripts.csv", index=False)
    
    print(f"\n✓ Created sample dataset with {len(df)} transcript segments")
    print(f"✓ Saved to: this_american_life_transcripts.csv")
    print(f"✓ Episodes: {df['episode_id'].nunique()}")
    print("\nSample data:")
    print(df.head())
    print("\nYou can now run: python setup_data.py")