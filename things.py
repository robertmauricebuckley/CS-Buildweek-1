"""Steps:
        Yes/no data:
        Find the first node
            -find the impurity score for each feature
                - using Gini impurity
        - first node (root of tree) will be the the feature with the best Gini score (lowest)
        - when lowest gini score is found:
            - instantiate new Node:

        - repeat process from remaining features untill current leaf's gini impurity score
        is less than any remaining options
        steps for finding value to split a node on:
        - order the values
        Numeric data:
        - find the average value of the adjecent values
                - ie 1) 8, average:9, 2) 10, average: 12, 3) 14
            - calculate impuirity value for each average value 
                - ie, weight < 12 
            - use the lower gini value from each of the averages to compare to the 
            other features' gini values
                - it will also become where we split the node if chosen as next
        Ranked data:
        - similar to numeric data, but don't find average this time
        - find gini for each rank value, except not the last one.
            - ie rank <= 1, rank <= 2, ...
        Multiple-choice data:
        - find gini score for all possible combination of choices.
            - ie choice: Blue, choice: Green, choice: Red, choice: Blue or Green
                choice, Blue or Red, choice: Green or Red.

        creating nodes/leaves:
            - features
                - name/value (feature),
                - left (name= True)
                - right (name = False)
                - gini score
                - number of samples that was fit to model
                - values of labeled y. array form [50, 50], which would be the results
                    from the split of previous node
            leaf:
                - features
                    - name/value (a labeled y value)
                    - gini score
                    - number of samples
                    - values of labeled y. array form. results from previous
                        node split.

        

        """


   """Make two methods
    MVP will be classification model
    1. fit
        - parameters that it needs to take in.
            - data
                - dataframes only
            - indicate features 
            - indicate labeled column

    2. predict
        - parameters that it needs to take in.
            - data
                - dataframes only
        - output
            - dataframe
            - single column
    """


    """Helper methods to be used in fit() method:
    """
    # def creat_root_node(self):
    #     """creat new instance of Node(),
    #     that become root/first node of tree
    #     """
    #     scores = []
    #     for i in x_columns:
    #         scores.append(find_gini(i,a_cls))

    #     index_num = scores.index(min(scores))
    #     self.node = Node(name='x_columns[index_num]', )

    # def add_leaf(self):
    #     """create new instance of Leaf()

    #     """
    #     pass
    
    # def change_to_node(self):
    #     """switch a leaf to a node.

    #     """
    #     pass






        # feature = data[[x_column, y_column]]
        # feature_true =  feature[feature[x_column] == split_value]
        # gini_score_t = 1
        # for i in (feature_true[y_column].unique()):
        # # print(i)
        #     gini_score_t -= ((feature_true[y_column].values == i).sum()/len(feature_true.index))**2

        # feature_false =  feature[feature[x_column] != split_value]
        # gini_score_f = 1
        # for i in (feature_false[y_column].unique()):
        # # print(i)
        #     gini_score_f -= ((feature_false[y_column].values == i).sum()/len(feature_false.index))**2

        # gini_score = ((len(feature_true.index)/(len(feature.index)))* gini_score_t) + ((len(feature_false.index)/(len(feature.index)))* gini_score_f)

        # return gini_score